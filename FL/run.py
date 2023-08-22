import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))

from typing import *
from helpers import *
from models.cnn import CNN
from client import FlowerNumPyClient
from strategies import FedAvg
from data_split import iid_split, non_iid_split
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.optim as optim
import flwr as fl
import wandb

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

NUM_CLIENTS = 10
NUM_ROUNDS = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def log_wandb():
    PROJECT_NAME = "federated-learning experiment"
    RUN = "server"
    config = {
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS,
    }
    wandb.login(key='cc7f4475483a016385fce422493eee957157cccd')
    wandb.init(
        project=PROJECT_NAME, name=RUN, 
        notes="A experiemnt on FL using Flower framework",
        config=config
    )

def load_datasets(num_clients: int, trigger: str, poison_ratio: float, poison_target: int, batch_size: int, iid: bool = True):
    # Download, transform, and split CIFAR-10 dataset among clients
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    
    trainset = CIFAR10("source/dataset/CIFAR10", train=True, download=True, transform=transform)
    testset = CIFAR10("source/dataset/CIFAR10", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    
    if iid:
        trainloaders, valloaders = iid_split(trainset, num_clients)
    else:
        trainloaders, valloaders = non_iid_split(trainset, num_clients)
        
    return trainloaders, valloaders, testloader

def client_fn(cid) -> FlowerNumPyClient:
    client_model = CNN().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    optimizer = optim.Adam(
        client_model.parameters(), weight_decay = 1e-4, 
        lr = 0.01, 
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max = NUM_ROUNDS 
    )
    return FlowerNumPyClient(cid, client_model, trainloader, valloader, lr_scheduler)

def fit_config(server_round: int):
    config = {
        "server_round": server_round,
        "local_epochs": 10,
    }
    return config


if __name__ == "__main__":
    # log_wandb()
    trigger='real_beard'
    poison_ratio=0.1
    poison_target=1
    batch_size=8
    trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS, trigger= trigger, poison_ratio= poison_ratio, poison_target= poison_target, batch_size= batch_size)
    print(trainloaders[0].dataset)
    # client_resources = None
    # if DEVICE.type == "cuda":
    #     client_resources = {"num_gpus": 1}

    # strategy = FedAvg(
    #     test_loader=testloader,
    #     server_model=CNN(),
    #     fraction_fit=0.5,  
    #     fraction_evaluate=0.5,  
    #     min_fit_clients=10,
    #     min_evaluate_clients=20,
    #     min_available_clients=NUM_CLIENTS,
    #     initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(CNN())),
    #     on_fit_config_fn=fit_config,
    # )
    
    # fl.simulation.start_simulation(
    #     client_fn=client_fn,
    #     num_clients=NUM_CLIENTS,
    #     config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS), 
    #     strategy=strategy,
    #     client_resources=client_resources,
    # )
    # wandb.finish()
