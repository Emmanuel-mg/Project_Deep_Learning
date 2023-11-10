import torch
import numpy as np

from PaiNN.data_loader import PaiNNDataLoader
from PaiNN.model import PaiNNModel
   
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
        self.loss = loss
        self.target = target
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
            batch = batch.to(self.device)
            targets = batch["y"][self.target]

            # Backpropagate using the selected loss
            outputs = self.model(batch)
            loss = self.loss(outputs, targets)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Cleanup at the end of the batch
            del batch
            del targets
            del loss
            del outputs
            torch.cuda.empty_cache()