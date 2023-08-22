from data_preprocessing import *
from helpers import *
from models.transfer_learning import ResNet, VGG16, DenseNet
from typing import *
from torchvision.datasets import ImageFolder
import os
from math import ceil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset, Dataset, Subset
import tqdm
import wandb

"""
Class mapping:
{'cuong': 0, 'dung': 1, 'khiem': 2, 'long': 3, 'nhan': 4, 'son': 5, 'thinh': 6, 'tuan': 7}
Hyperparameter for physical backdoor attack:
trigger type: One of {'real_beard', 'real_beard+red_hat', 'black_earings', 'yellow_hat', 'red_hat', 'sunglasses', 'big_yellow_earings', 'white_face_mask', 'fake_beard', 'blue_face_mask', 'big_sticker', 'sunglasses+hat', 'yellow_sticker', 'white_earings', 'blue_sticker', 'small_earings', 'black_face_mask', 'additional_image'}
poison_ratio: the percentage of poisoned data in the training set
poison_label_num: the number of classes that have poisoned data
poison_target: the target label of the poisoned data
"""

# Addition_image trigger is not in trigger_list
trigger_list = {'real_beard', 'black_earings', 'yellow_hat', 'red_hat', 'sunglasses', 'big_yellow_earings', 'white_face_mask', 'fake_beard', 'blue_face_mask', 'big_sticker', 'sunglasses+hat', 'yellow_sticker', 'white_earings', 'blue_sticker', 'small_earings', 'black_face_mask'}    
# Code reference: https://discuss.pytorch.org/t/using-weightedrandomsampler-with-concatdataset/51968/2
def weighted_sampler(data: Dataset, num_classes: int) -> WeightedRandomSampler:
    """
    This function returns WeightedRandomSampler to solve Data Imbalance
    Params:
        data: Traning Dataset
    Returns:
        WeightedRandomSampler
    """
    
    # Get the targets of the dataset
    targets = []
    for _, target in data:
        targets.append(target)
    targets = torch.tensor(targets)

    # Compute samples weight (each sample should get its own weight)
    class_sample_count = torch.tensor([(targets == t).sum() for t in range(num_classes)])
    weight = [torch.tensor(0) for i in range(num_classes)]
    for i in range(num_classes):
        if class_sample_count[i] != 0:
            weight[i] = 1.0 / class_sample_count[i].float()
    samples_weight = torch.tensor([weight[t] for t in targets])

    # return sampler
    return WeightedRandomSampler(samples_weight, len(samples_weight))

def load_data(src_dir: str, trigger: str, poison_ratio: float, poison_target: int, batch_size: int) -> Tuple[Set[Dataset], Set[DataLoader]]:
    clean_ds_train = ImageFolder(root=os.path.join(src_dir, 'clean_image', 'train'), transform=data_transforms['train_augment'], loader=loader)
    clean_ds_test = ImageFolder(root=os.path.join(src_dir, 'clean_image', 'test'), transform=data_transforms['test'])
    class_mapping = clean_ds_train.class_to_idx

    assert trigger in trigger_list, f"{trigger} is not in the trigger list"
    assert poison_target in class_mapping.values(), f"{poison_target} is not a valid target label"
    assert poison_ratio <= len(class_mapping), f"Number of poison labels is greater than total number of labels({len(class_mapping)})" 
    assert poison_ratio >= 0 and poison_ratio <= 1, "Poison ratio must be >= 0 and <= 1"

    poison_ds_train = ImageFolder(root=os.path.join(src_dir, trigger, 'train'), loader=loader)
    poison_ds_test = ImageFolder(root=os.path.join(src_dir, trigger, 'test'), transform=data_transforms['test'])

    # Test screnario 1: Sample (num_clean * poison_ratio/1-poison_ratio) images from the trigger directory. 
    num_clean_images = len(clean_ds_train)
    num_flipped_images = ceil(poison_ratio/(1-poison_ratio) * num_clean_images)
    num_poison_images = len(poison_ds_train)
    assert num_poison_images <= len(poison_ds_train), "Poison ratio is too high"

    # Sample num_flipped_images from the trigger directory such that the indices from target class are not sampled
    indices_to_sample = [i for i in range(0, num_poison_images) if poison_ds_train[i][1] != poison_target]
    poison_indices = set(random.sample(indices_to_sample, num_flipped_images)) 
    remaining_indices = set(range(0, num_poison_images)) - poison_indices

    # Update dataset. Poison test set will include (remaining_images + images in the test folder)
    poison_indices = sorted(list(poison_indices))
    remaining_indices = sorted(list(remaining_indices))
    
    remain_dataset = CustomizedDataset(Subset(dataset=poison_ds_train, indices=remaining_indices), transform=data_transforms['test'])
    poison_ds_test = ConcatDataset([poison_ds_test, remain_dataset])
    
    poison_ds_train = CustomizedDataset(PoisonSubset(dataset=poison_ds_train, indices=poison_indices, poison_target=poison_target), transform=data_transforms['train_augment'])
    train_dataset = ConcatDataset([clean_ds_train, poison_ds_train])

    datasets = {
        'train': train_dataset,
        'clean_test': clean_ds_test,
        'poison_test': poison_ds_test 
    }

    dataloaders = {
        'train': DataLoader(datasets['train'],
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=6,
                                sampler=weighted_sampler(datasets['train'], num_classes=len(clean_ds_train.classes)),
                                worker_init_fn=worker_init_fn),  
        'clean_test': DataLoader(datasets['clean_test'],
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                worker_init_fn=worker_init_fn), 
        'poison_test': DataLoader(datasets['poison_test'],
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                worker_init_fn=worker_init_fn) 
    }
    return datasets, dataloaders

def train_model(model, criterion, optimizer, scheduler, num_epochs=20, runs=3):
    train_acc = [[] for i in range(num_epochs)]
    train_loss = [[] for i in range(num_epochs)]
    clean_acc = [[] for i in range(num_epochs)]
    clean_loss = [[] for i in range(num_epochs)]
    attack_success_rate = [[] for i in range(num_epochs)]

    for run in range(runs):
        model.reset_weights()
        print(f"\nRun {run+1}:\n")
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 60)
            for phase in ['train', 'clean_test', 'poison_test']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm.tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    if phase == 'poison_test':
                        running_corrects += torch.sum(preds == poison_target)
                    else:
                        running_corrects += torch.sum(preds == labels.data)
                    
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / len(datasets[phase])
                epoch_acc = running_corrects.double() / len(datasets[phase])

                if phase == 'poison_test':
                    print('Attack success rate {:.4f}'.format(epoch_acc))
                    attack_success_rate[epoch].append(epoch_acc.item())
                else:                                  
                    print('{} loss: {:.4f}, {} acc: {:.4f}'.format(phase.capitalize(),
                                                                    epoch_loss,
                                                                    phase.capitalize(),
                                                                    epoch_acc))
                    if phase == "clean_test": 
                        clean_acc[epoch].append(epoch_acc.item())
                        clean_loss[epoch].append(epoch_loss)
                    elif phase == "train": 
                        train_acc[epoch].append(epoch_acc.item())
                        train_loss[epoch].append(epoch_loss)
                        
    print("\n" + "Average results after 3 runs:")
    for epoch in range(num_epochs):
        tr_acc = round(sum(train_acc[epoch]) / runs, 2)
        tr_loss = round(sum(train_loss[epoch]) / runs, 2)
        c_acc = round(sum(clean_acc[epoch]) / runs, 2)
        c_loss = round(sum(clean_loss[epoch]) / runs, 2)
        asr = round(sum(attack_success_rate[epoch]) / runs, 2)
        
        print(f"Epoch {epoch + 1}:")
        print(f"Train accuracy: {tr_acc}; Train loss: {tr_loss}; \n"
            f"Clean accuracy: {c_acc}; Clean loss: {c_loss}; \n"
            f"Attack success rate: {asr} ")
        print("-"*60)
        
        wandb.log({"Training accuracy": tr_acc,
                    "Training loss": tr_loss,
                    "Clean accuracy": c_acc,
                    "Clean loss": c_loss,
                    "Attack success rate": asr
                    }, step = epoch+1)    
        
    table.add_data(trigger, c_acc, asr)

def run(num_epochs, trigger):
    print("-" * 10 + f"Physical backdoor with {trigger} trigger on {model_name}" + "-" * 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lr=0.001, params=model.parameters())
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)
    print("\n")

def log_wandb(trigger, poison_target, poison_ratio, model_name, num_epochs):
    PROJECT_NAME = f"Physical Backdoor in Centralised Setting with Transfer_Learning ({model_name})"
    RUN = trigger
    wandb.init(
        project=PROJECT_NAME, name=RUN, 
        notes=f"model: {model_name} (Transfer Learning); poison_rate: {poison_ratio} (randomly sampled); poison_target: {poison_target}; batch_size: {batch_size}; epochs: {num_epochs}."
    )
    cfg = wandb.config
    cfg.update({
        "trigger": trigger, "poison_target": poison_target, "poison_ratio": poison_ratio, 
        "model_name": model_name, "batch_size": batch_size, "num_epochs": num_epochs, "runs": runs,
        "criterion": "Cross Entropy Loss", "optimizer": "Adam", "lr": 0.001, "scheduler": "StepLR(step_size=5,gamma=0.1)" 
    })

if __name__ == "__main__":
    wandb.login(key='cc7f4475483a016385fce422493eee957157cccd')
    src_dir = '/vinserver_user/21thinh.dd/FedBackdoor/source/dataset/facial_recognition_rescale_split'
    trigger='real_beard'
    poison_ratio=0.1
    poison_target=1
    batch_size=8
    num_epochs=10
    runs=3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    models = [ResNet(num_classes=8).to(device), VGG16(num_classes=8).to(device), DenseNet(num_classes=8).to(device)]
    for i in range(len(models)):
        model = models[i]
        model_name = str(model)
        for trigger in trigger_list:
            log_wandb(trigger, poison_target, poison_ratio, model_name, num_epochs)
            table = wandb.Table(columns=['trigger', 'clean accuracy', 'attack success rate'])
            datasets, dataloaders = load_data(src_dir=src_dir, trigger=trigger, poison_ratio=poison_ratio, poison_target=poison_target, batch_size=batch_size)
            run(num_epochs, trigger)
            wandb.finish()