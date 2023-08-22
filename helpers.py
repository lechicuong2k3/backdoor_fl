from typing import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import tqdm

"""
Contain helper functions for training and evaluating
"""

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net: nn.Module, parameters: List[np.ndarray]):
    """Set parameters for the model"""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train(net: nn.Module, trainloader: DataLoader, epochs: int) -> None:
    """Train the network on the training set."""
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(net: nn.Module, testloader: DataLoader):
    """Evaluate the network on the entire test set."""
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in tqdm.tqdm(testloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def additional_metrics(net, testloader, target_label):
    """Contain attack_success_rate (ASR), target_class_accuracy (TCA)"""
    pass


class CustomizedDataset(Dataset):
    r"""
    A CustomizedDataset convert a Subset object to a Dataset and apply the transform as defined.

    Args:
        subset (Subset): The converted Subset object
        transform (torchvision.transforms.Compose): Transform object
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

# Code reference: https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209/4
class PoisonSubset(Dataset):
    r"""
    A Poison Subset of a dataset at specified indices, where the labels of all samples are flipped to target label.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: Dataset, indices: list[int], poison_target) -> None:
        self.dataset = dataset
        self.indices = indices
        self.poison_target = poison_target

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]][0], self.poison_target

    def __len__(self):
        return len(self.indices)