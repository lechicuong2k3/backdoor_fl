from helpers import *
import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
import wandb

class FedAvg(fl.server.strategy.FedAvg):
    def __init__(self, 
        test_loader, 
        server_model, 
        *args, **kwargs, 
    ):
        self.test_loader = test_loader
        self.server_model = server_model
        super().__init__(*args, **kwargs)
    
    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("-"*40 + "Start Server Testing" + "-"*40)
        self.server_model = self.server_model.to(device)
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        set_parameters(self.server_model, parameters_ndarrays)  # Update model with the latest parameters
        loss, test_accuracy = test(self.server_model, self.test_loader)
        print(f"Server-side evaluation loss {loss} / accuracy {test_accuracy}")
        wandb.log({"test_loss": loss, "test_accuracy": test_accuracy}, step = server_round)
        
        print("-"*40 + "Finish Server Testing" + "-"*40)
        return loss, {"accuracy": test_accuracy}

