import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))

from data_preprocessing import *
from typing import *
from torchvision.datasets import ImageFolder
from math import ceil
import random
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset, Dataset, Subset
import wandb
from helpers import *
from models.cnn import CNN
from client import FlowerNumPyClient
from strategies import FedAvg
from data_split import iid_split, non_iid_split
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import flwr as fl
import wandb

# Addition_image trigger is not in trigger_list
trigger_list = {'real_beard', 'black_earings', 'yellow_hat', 'red_hat', 'sunglasses', 'big_yellow_earings', 'white_face_mask', 'fake_beard', 'blue_face_mask', 'big_sticker', 'sunglasses+hat', 'yellow_sticker', 'white_earings', 'blue_sticker', 'small_earings', 'black_face_mask'}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

NUM_CLIENTS = 10
NUM_ROUNDS = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(trigger: str, poison_ratio: float, poison_target: int, batch_size: int, iid: bool = True) -> Tuple[Set[Dataset], Set[DataLoader]]:
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


    num_classes = 8
    if iid:
        trainloaders, valloaders = iid_split(train_dataset, num_classes, NUM_CLIENTS)
    else:
        trainloaders, valloaders = non_iid_split(train_dataset, num_classes, NUM_CLIENTS, balance = 0.1)
        
    datasets = {
        'train': train_dataset,
        'clean_test': clean_ds_test,
        'poison_test': poison_ds_test 
    }

    dataloaders = {
        'train': trainloaders,
        'validation': valloaders,
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

def log_wandb(trigger, poison_target, poison_ratio, model_name, num_epochs):
    PROJECT_NAME = "Physical Backdoor in Centralised Setting (Resnet50-Transfer_Learning) (fixed bug)"
    RUN = trigger
    wandb.login(key='cc7f4475483a016385fce422493eee957157cccd')
    wandb.init(
        project=PROJECT_NAME, name=RUN, 
        notes=f"model: {model_name} (Transfer Learning); poison_rate: {poison_ratio} (randomly sampled); poison_target: {poison_target}; batch_size: {batch_size}; epochs: {num_epochs}."
    )

if __name__ == "__main__":
    trigger='real_beard'
    poison_ratio=0.1
    poison_target=1
    model_name="ResNet50"
    batch_size=8
    num_epochs=30
    src_dir = '/vinserver_user/21thinh.dd/FedBackdoor/source/dataset/facial_recognition_rescale_split_augment'
    datasets, dataloaders = load_data(trigger=trigger, poison_ratio=poison_ratio, poison_target=poison_target, batch_size=batch_size, iid= True)
