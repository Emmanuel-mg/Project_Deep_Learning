import torch
import numpy as np

from PaiNN.data_loader import PaiNNDataLoader
from PaiNN.model import PaiNNModel
from PaiNN.utils import mse

class Trainer:
    """ Responsible for training loop and validation """
    
    def __init__(self, model: torch.nn.Module, loss: any, target: int, optimizer: torch.optim, data_loader, device: torch.device):
        """ Constructor
        Args:   
            model: Model to use (usually PaiNN)
            loss: loss function to use during traning
            target: the index of the target we want to predict 
            optimizer: optimizer to use during training
            data_loader: DataLoader object containing train/val/test sets
            device: device on which to execute the training
        """
        self.model = model
        self.target = target
        self.loss = loss
        self.optimizer = optimizer

        self.train_set = data_loader
        self.valid_set = data_loader.get_val()
        self.test_set = data_loader.get_test()
        self.device = device

    def _train_epoch(self) -> dict:
        """ Training logic for an epoch
        """
        for batch_idx, batch in enumerate(self.train_set):
            # Using our chosen device
            batch = batch
            targets = batch["targets"][:, self.target].to(self.device)
            print(targets.shape)
            # Backpropagate using the selected loss
            outputs = self.model(batch).to(self.device)
            loss = self.loss(outputs, targets)

            print(f"Current loss {loss} Current batch {batch_idx}")

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Cleanup at the end of the batch
            del batch
            del targets
            del loss
            del outputs
            torch.cuda.empty_cache()

if __name__=="__main__":
    with torch.autograd.set_detect_anomaly(True):
        train_set = PaiNNDataLoader(r_cut=2, batch_size=32)
        model = PaiNNModel(r_cut=2)
        optimizer = torch.optim.Adam(params=model.parameters())
        trainer = Trainer(
            model=model,
            loss=mse,
            target=0,
            optimizer=optimizer,
            data_loader=train_set,
            device="cpu"
        )
        trainer._train_epoch() 