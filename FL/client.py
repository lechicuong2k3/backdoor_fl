import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))

from helpers import *
import wandb 
import flwr as fl

import warnings
warnings.simplefilter("ignore", UserWarning) # Ignore warning for lr_scheduler

class FlowerNumPyClient(fl.client.NumPyClient):
    def __init__(self, 
                cid, net, 
                trainloader, 
                valloader, 
                lr_scheduler):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.lr_scheduler = lr_scheduler

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        num_epochs = config["local_epochs"]      
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=num_epochs) 
        self.lr_scheduler.step()  
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        wandb.log({"evaluate_loss":loss, "evaluate_accuracy": accuracy}, step = self.round)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

