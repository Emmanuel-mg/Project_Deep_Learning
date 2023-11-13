import torch
import numpy as np

from PaiNN.data_loader import PaiNNDataLoader
from PaiNN.model import PaiNNModel
from PaiNN.utils import mse

class Trainer:
    """ Responsible for training loop and validation """
    
    def __init__(self, model: torch.nn.Module, loss: any, target: int, optimizer: torch.optim, data_loader, scheduler: torch.optim, device: torch.device):
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
        self.device = device
        self.learning_rate = []
        self.valid_perf= []
        self.test_perf = []
        self.summaries, self.summaries_axes = plt.subplots(1,3, figsize=(10,5))


    def _train_epoch(self) -> dict:
        """ Training logic for an epoch
        """
        for batch_idx, batch in enumerate(self.train_set):
            # Using our chosen device
            targets = batch["targets"][:, self.target].to(self.device)

            # Backpropagate using the selected loss
            outputs = self.model(batch).to(self.device)
            loss = self.loss(outputs, targets)

            print(f"Current loss {loss} Current batch {batch_idx}")

            self.learning_rate.append(loss.item())

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

                self.test_perf.append(self.loss(pred_val, targets).item())
                
                del targets
                del pred_val

        return val_loss/(batch_idx+1)

    def plot_data(self):
        p_data = (self.learning_rate, self.valid_perf, self.test_perf)
        plot_names = ['Learning curve','Validation loss for every 400 batches', 'Evaluation loss']

        for i in range(3):
            self.summaries_axes[i].plot(p_data[i])
            self.summaries_axes[i].set_ylabel('Loss')
            self.summaries_axes[i].set_xlabel('Batches')
            self.summaries_axes[i].set_xlim((0, len(self.train_set)))
            self.summaries_axes[i].set_title(plot_names[i])

        x_ticks = np.arange(0, len(self.train_set), 400)
        x_ticks_label = [str(val) for val in x_ticks]
        self.summaries_axes[1].set_xticks(x_ticks)
        self.summeries_axes[1].set_xticklabels(x_ticks_label)
        plt.savefig('Loss_plot', dpi=800)
        plt.show()


        