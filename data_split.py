"""
This file contains functions to split datasets in different ways
"""
import torch
from torch.utils.data import Subset, Dataset, random_split, DataLoader, WeightedRandomSampler
import numpy as np

# Code reference: https://discuss.pytorch.org/t/using-weightedrandomsampler-with-concatdataset/51968/2
def weighted_sampler(data: Dataset, num_classes) -> WeightedRandomSampler:
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

def iid_split(train_dataset: Dataset, num_classes: int, num_clients: int) -> tuple[list[DataLoader], list[DataLoader]]:
    """
    Return an array of Datasets split in IID fashion
    """
    validation_split = 0.1
    train_num_data = len(train_dataset)

    train_dict_users = balanced_split(train_num_data, num_classes, num_clients)
    train_loaders = []
    val_loaders = []
    for client in train_dict_users.keys():
        train_data = Subset(train_dataset, list(train_dict_users[client]))
        len_val = int(len(train_data) * validation_split)
        len_train = len(train_data) - len_val
        lengths = [len_train, len_val]
            
        train_data, val_data = random_split(train_data, lengths, torch.Generator().manual_seed(42))
        train_loaders.append(DataLoader(train_data, batch_size=8, shuffle=False, num_workers=6, sampler=weighted_sampler(train_data, num_classes)))
        val_loaders.append(DataLoader(val_data, batch_size=8, num_workers=4, shuffle=False))

    
    return train_loaders, val_loaders

def non_iid_split(train_dataset: Dataset, num_classes: int, num_clients: int, balance : float) -> tuple[list[DataLoader], list[DataLoader]]:
    """
    Return an array of Datasets split in IID fashion
    """
    validation_split = 0.1
    train_num_data = len(train_dataset)
    
    train_dict_users = imbalanced_split(train_num_data, num_classes, num_clients, balance)
    train_loaders = []
    val_loaders = []
    for client in train_dict_users.keys():
        train_data = Subset(train_dataset, list(train_dict_users[client]))
        len_val = int(len(train_data) * validation_split)
        len_train = len(train_data) - len_val
        lengths = [len_train, len_val]
        
        train_data, val_data = random_split(train_data, lengths, torch.Generator().manual_seed(42))
        train_loaders.append(DataLoader(train_data, batch_size=8, shuffle=False, num_workers=6,sampler=weighted_sampler(train_data, num_classes)))
        val_loaders.append(DataLoader(val_data, batch_size=8, num_workers=4, shuffle=False))

    
    return train_loaders, val_loaders

def balanced_split(num_data: int, num_classes: int, num_users: int):
    nitems_per_class = int(num_data/num_classes)
    dict_users = dict()
    for i in range(num_users):
        dict_users[i] = set()

    dict_items = dict()
    for i in range(num_classes):
        idx = nitems_per_class * i
        dict_items[i] = set(range(idx, nitems_per_class + idx))

    data_split = {}
    for label in range(num_classes):
        distribution = np.ones(num_users) * 1/num_users
        data_split[label] = np.round(distribution * nitems_per_class).astype(int)
        if (sum(data_split[label]) < nitems_per_class):
            for i in range(nitems_per_class - sum(data_split[label])):
                user = np.random.randint(0, num_users)
                data_split[label][user] += 1
        elif (sum(data_split[label]) > nitems_per_class):
            for i in range(sum(data_split[label]) - nitems_per_class):
                user = np.random.randint(0, num_users)
                data_split[label][user] -= 1
    
    for i in range(num_users):
        for j in range(num_classes):    
            addition = set(np.random.choice(list(dict_items[j]), data_split[j][i], replace=False))    
            dict_users[i] |= addition
            dict_items[j] -= addition
    
    return dict_users

def imbalanced_split(num_data: int, num_classes: int, num_users: int,  balance: float) -> dict[str, list[int]]:
    """
    balance controls the imbalance rate of data split
    """
    nitems_per_class = int(num_data/num_classes)
    
    dict_users = dict()
    for i in range(num_users):
        dict_users[i] = set()

    dict_items = dict()
    for i in range(num_classes):
        idx = nitems_per_class * i
        dict_items[i] = set(range(idx, nitems_per_class + idx))

    data_split = {}
    for label in range(num_classes):
        distribution = np.random.dirichlet(np.full(num_users, balance))
        data_split[label] = np.round(distribution * nitems_per_class).astype(int)
        if (sum(data_split[label]) < nitems_per_class):
            for i in range(nitems_per_class - sum(data_split[label])):
                user = np.random.randint(0, num_users)
                data_split[label][user] += 1
        elif (sum(data_split[label]) > nitems_per_class):
            for i in range(sum(data_split[label]) - nitems_per_class):
                user = np.random.randint(0, num_users)
                while data_split[label][user] == 0:
                    user = np.random.randint(0, num_users)
                data_split[label][user] -= 1
    
    for i in range(num_users):
        for j in range(num_classes):    
            addition = set(np.random.choice(list(dict_items[j]), data_split[j][i], replace=False))    
            dict_users[i] |= addition
            dict_items[j] -= addition
    
    return dict_users

def plot_data(self, dict_users, dataset, partition_type, train):
    import matplotlib.pyplot as plt
    clients = ['Client ' + str(i) for i in range(1, self.num_users + 1)]
    client_class_counts = [[0 for i in range(self.num_users)] for j in range(self.num_classes)]
    if self.dataset == 'cifar10':
        clients = ['Client ' + str(i) for i in range(1, self.num_users + 1)]
        for user in range(self.num_users):
            for index in dict_users[user]:
                if train:
                    label = index//5000
                else:
                    label = index//1000
                client_class_counts[label][user] += 1
                
    elif self.dataset == 'cifar100':
        for user in range(self.num_users):
            for index in dict_users[user]:
                if train:
                    label = index//500
                else:
                    label = index//100
                client_class_counts[label][user] += 1

    elif self.dataset == 'mnist':
        for user in range(self.num_users):
            for index in dict_users[user]:
                if train:
                    label, _ = self.mnist_label(index, train=True)
                else:
                    label, _ = self.mnist_label(index, train=False)
                client_class_counts[int(label)][user] += 1
        
    client_class_counts = np.array(client_class_counts)
    fig = plt.subplots(figsize =(15, 12))
    bottom = np.zeros(len(clients))

    for i, class_name in enumerate(self.labels):
        plt.bar(clients, client_class_counts[i], bottom=bottom, label=class_name, width=0.7)
        bottom += client_class_counts[i]

    plt.xlabel('Clients')
    plt.ylabel('Number of Data')
    if train:
        plt.title(f'Train Data Partition for {self.num_users} clients ({self.dataset.upper()} Dataset)')
    else:
        plt.title(f'Test Data Partition for {self.num_users} clients ({self.dataset.upper()} Dataset)')
    # plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if train:
        plt.savefig(f'{self.dataset}_train_{partition_type}.png')
    else:
        plt.savefig(f'{self.dataset}_test_{partition_type}.png')