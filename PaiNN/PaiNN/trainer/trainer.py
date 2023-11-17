import torch
import numpy as np
import matplotlib.pyplot as plt

from PaiNN.data_loader import PaiNNDataLoader
from PaiNN.model import PaiNNModel
from PaiNN.utils import mse

class Trainer:
    """ Responsible for training loop and validation """
    
    def __init__(self, model: torch.nn.Module, loss: any, target: int, optimizer: torch.optim, data_loader, scheduler: torch.optim):
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
        self.scheduler = scheduler

        self.train_set = data_loader
        self.valid_set = data_loader.get_val()
        self.test_set = data_loader.get_test()
        self.learning_curve = []
        self.valid_perf= []
        self.learning_rates = []
        self.summaries, self.summaries_axes = plt.subplots(1,3, figsize=(10,5))


    def _train_epoch(self) -> dict:
        """ Training logic for an epoch
        """
        for batch_idx, batch in enumerate(self.train_set):
            # Using our chosen device
            targets = batch["targets"][:, self.target].to(self.device)

            # Backpropagate using the selected loss
            outputs = self.model(batch)
            loss = self.loss(outputs, targets)

            print(f"Current loss {loss} Current batch {batch_idx}")

            self.learning_curve.append(loss.item())
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if (batch_idx+1)%400 == 0:
                val_loss = self._eval_model()
                print(f"Validation loss for {batch_idx} is {val_loss.item()}")
                self.scheduler.step(val_loss)
                self.valid_perf.append(val_loss.item())

                del val_loss

            # Cleanup at the end of the batch
            del batch
            del targets
            del loss
            del outputs
            torch.cuda.empty_cache()

    def _eval_model(self):
        val_loss = torch.zeros(1)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_set):
                pred_val = self.model(batch).to(self.device)
                targets = batch["targets"][:, self.target].to(self.device)
                
                val_loss = val_loss + self.loss(pred_val, targets)
                
                del targets
                del pred_val

        return val_loss/(batch_idx+1)

    def _train(self, num_epoch: int = 10):
        """ Method to train the model
        Args:
            num_epoch: number of epochs you want to train for
        """
        for _ in range(num_epoch):
            self._train_epoch()
        
    def plot_data(self):
        p_data = (self.learning_curve, self.valid_perf, self.learning_rates)
        plot_names = ['Learning curve','Validation loss for every 400 batches', 'Learning rates']

        for i in range(3):
            self.summaries_axes[i].plot(range(len(p_data[i])), p_data[i])
            self.summaries_axes[i].set_ylabel('Loss')
            self.summaries_axes[i].set_xlabel('Batches')
            self.summaries_axes[i].set_xlim((0, len(p_data[i])))
            self.summaries_axes[i].set_title(plot_names[i])

        plt.savefig('Loss_plot.png', dpi=800)
        plt.show()


        