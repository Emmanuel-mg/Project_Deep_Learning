import torch
import numpy as np
import matplotlib.pyplot as plt

from PaiNN.data_loader import PaiNNDataLoader
from PaiNN.model import PaiNNModel
from PaiNN.utils import mse

class Trainer:
    """ Responsible for training loop and validation """
    
    def __init__(self, model: torch.nn.Module, loss: any, metric: any, target: int, optimizer: torch.optim, data_loader, scheduler: torch.optim, device: torch.device = "cpu"):
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
        self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.train_set = data_loader
        self.mean, self.std = self.standardize_data()
        self.valid_set = data_loader.get_val()
        self.test_set = data_loader.get_test()
        self.epoch_swa = 100
        self.learning_curve = []
        self.valid_curve = []
        self.valid_loss = []
        self.learning_rates = []
        self.summaries, self.summaries_axes = plt.subplots(1,3, figsize=(10,5))


    def _train_epoch(self) -> dict:
        """ Training logic for an epoch
        """
        for batch_idx, batch in enumerate(self.train_set):

            mean_loss = torch.zeros(1).to(self.device)
            mean_mae = torch.zeros(1).to(self.device)
            # Using our chosen device
            targets = batch["targets"][:, self.target].to(self.device).unsqueeze(dim=-1)
            # Standardizing the data
            targets = (targets - self.mean[self.target])/self.std[self.target]  

            # Backpropagate using the selected loss
            outputs = self.model(batch)
            loss = self.loss(outputs, targets)

            # Tracking the results of the epoch
            mean_loss = mean_loss + loss
            mean_mae = mean_mae + self.metric(outputs*self.std[self.target] + self.mean[self.target], 
                                                    targets*self.std[self.target] + self.mean[self.target])

            # Tracking loss during training
            if batch_idx%100 == 0:
                print(f"Current loss {mean_loss/(batch_idx+1)} Current batch {batch_idx}/{len(self.train_set)} ({100*batch_idx/len(self.train_set):.2f}%)")

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Cleanup at the end of the batch
            del batch
            del targets
            del loss
            del outputs
            torch.cuda.empty_cache()
        
        # Printing the result of the epoch 
        if self.target not in [0, 1, 5, 11, 16, 17, 18]:
            mean_mae = mean_mae * 1000
        print("MAE for the training set (last batch)", mean_mae.item()/(batch_idx + 1))

        # Tracking results for plotting
        self.learning_curve.append(mean_mae.item()/(batch_idx + 1))
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)

    def _eval_model(self):
        val_loss = torch.zeros(1).to(self.device)
        val_metric = torch.zeros(1).to(self.device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_set):
                pred_val = self.model(batch)
                targets = batch["targets"][:, self.target].to(self.device).unsqueeze(dim=-1)
                targets = (targets - self.mean[self.target])/self.std[self.target] 

                val_loss = val_loss + self.loss(pred_val, targets)
                # De-standardize the data
                val_metric = val_metric + self.metric(pred_val*self.std[self.target] + self.mean[self.target],
                                                       targets*self.std[self.target] + self.mean[self.target])
                
                del targets
                del pred_val

        # Convert units if necessary
        if self.target not in [0, 1, 5, 11, 16, 17, 18]:
            val_metric = val_metric * 1000

        return val_loss/(batch_idx+1), val_metric/(batch_idx+1)

    def _train(self, num_epoch: int = 10, early_stopping: int = 30, alpha: float = 0.9):
        """ Method to train the model
        Args:
            num_epoch: number of epochs you want to train for
            alpha: exponential smoothing factor
        """
        patience = 0
        for epoch in range(num_epoch):
            self._train_epoch()
            # Validate at the end of an epoch
            val_loss, val_metric = self._eval_model()

            # Tracking MAE in the validation
            self.valid_curve.append(val_metric.item())
            print(f"### End of the epoch : Validation loss for {epoch} is {val_loss.item()}")
            print(f"MAE for the validation set is {val_metric.item()}")

            # Exponential smoothing for validation
            val_loss_s = val_loss.item()
            self.valid_loss.append(val_loss_s if epoch == 0 else alpha*val_loss_s + (1-alpha)*self.valid_loss[-1])
            # LR scheduler (reduce on plateau)
            if epoch < self.epoch_swa:
                self.scheduler.step(self.valid_loss[-1])
            

            # Early stopping
            if epoch != 0 and min(min_loss, val_loss_s) == min_loss:
                patience +=1
                if patience >= early_stopping:
                    break
            else:
                patience = 0
            min_loss = val_loss_s if epoch == 0 else min(min_loss, val_loss_s)

            # Cleaning the GPU
            del val_loss        

    def standardize_data(self):
        """ Calculate means and standard deviations
        """
        train_set = []
        for _, batch in enumerate(self.train_set):
            train_set.append(batch['targets'])
        train_set = torch.cat(train_set).squeeze(dim=1)

        return torch.mean(train_set, axis=-2), torch.std(train_set, axis=-2)

    def plot_data(self):
        p_data = (self.learning_curve, self.valid_curve, self.learning_rates)
        plot_names = ['Learning curve','Validation loss for every 400 batches', 'Learning rates']

        for i in range(3):
            self.summaries_axes[i].plot(range(len(p_data[i])), p_data[i])
            self.summaries_axes[i].set_ylabel('Loss')
            self.summaries_axes[i].set_xlabel('Epochs')
            self.summaries_axes[i].set_xlim((0, len(p_data[i])))
            self.summaries_axes[i].set_title(plot_names[i])

        plt.savefig('Loss_plot.png', dpi=800)
        plt.show()


        