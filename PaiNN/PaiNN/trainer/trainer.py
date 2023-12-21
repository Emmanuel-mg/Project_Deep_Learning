import torch
import matplotlib.pyplot as plt

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
        self.learning_curve = []
        self.valid_curve = []
        self.valid_loss = []
        self.learning_rates = []
        self.summaries, self.summaries_axes = plt.subplots(1,3, figsize=(10,5))


    def _train_epoch(self) -> dict:
        """ Training logic for an epoch
        """
        mean_loss = torch.zeros(1).to(self.device)
        mean_mae = torch.zeros(1).to(self.device)

        for batch_idx, batch in enumerate(self.train_set):

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
                print(f"Current loss {mean_loss.item()/(batch_idx+1)} Current batch {batch_idx}/{len(self.train_set)} ({100*batch_idx/len(self.train_set):.2f}%)")

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
        print("MAE for the training set", mean_mae.item()/(batch_idx + 1))

        # Tracking results for plotting
        self.learning_curve.append(mean_mae.item()/(batch_idx + 1))
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)

    def _train(self, num_epoch: int = 100, epoch_swa: int = 100, early_stopping: int = 0, alpha: float = 0.9, acyclical: bool = True):
        """ Method to train the model
        Args:
            num_epoch: number of epochs you want to train for
            epoch_swa: epoch index at which we begin the SWA process
            early_stopping: if specified patience at which time we stop the training process
            alpha: exponential smoothing factor for the validation loss (influence on patience)
            acyclical : wether we apply cyclical or acyclical SWA
        """
        patience = 0
        swa_step = 0
        for epoch in range(num_epoch):
            # Test wether or not we should apply SWA procedure
            if epoch < epoch_swa:
                self._train_epoch()
                last_lr = self.learning_rates[-1]
                last_weights = self.model.get_weights()
            else:
                if acyclical:
                    swa_step += 1
                    self._train_epoch_swa_acyclical(weights = last_weights, step = swa_step, alpha = last_lr)     
                else:    
                    self._train_epoch_swa(alpha_1 = last_lr*10, alpha_2 = last_lr, c = 100)     

            # Validate at the end of an epoch (SWA or not SWA)
            val_loss, val_metric = self._eval_model()

            # Tracking MAE in the validation
            self.valid_curve.append(val_metric.item())
            print(f"### End of the epoch : Validation loss for {epoch} is {val_loss.item()}")
            print(f"MAE for the validation set is {val_metric.item()}")

            # Exponential smoothing for validation (factor defined by the user)
            val_loss_s = val_loss.item()
            self.valid_loss.append(val_loss_s if epoch == 0 else alpha*val_loss_s + (1-alpha)*self.valid_loss[-1])
            
            # LR scheduler (reduce on plateau as a baseline) if not SWA 
            if epoch < epoch_swa:
                self.scheduler.step(self.valid_loss[-1])
            # if SWA has begun the LR scheduler isn't updated anymore instead we handle it in the SWA procedure

            # Early stopping (if defined )
            if early_stopping!=0:
                if epoch != 0 and min(min_loss, val_loss_s) == min_loss:
                    patience +=1
                    if patience >= early_stopping:
                        break
                else:
                    patience = 0
                min_loss = val_loss_s if epoch == 0 else min(min_loss, val_loss_s)

            # Cleaning the GPU
            del val_loss      

    def _train_epoch_swa(self, alpha_1: float = 0.005, alpha_2: float = 0.001, c: int = 3) -> dict:
        """ Training logic for an epoch with Stochastic Weight Averaging with cycles over batches 
        Args:
            alpha_1: top learning rate of the cycle
            alpha_2: bottom learning rate of the cycle
            c: number of learning rates in the cycle
        """
        mean_loss = torch.zeros(1).to(self.device)
        mean_mae = torch.zeros(1).to(self.device)

        swa_weights = self.model.get_weights()
        swa_learning_rates = []
        swa_loss = []
        swa_metric = []

        for batch_idx, batch in enumerate(self.train_set):
            # Update the learning rate according to the schedule
            self.optimizer.param_groups[0]['lr'] = alpha_1 * (1 - (batch_idx % c)/(c - 1)) + alpha_2 * (batch_idx % c)/(c - 1)
            swa_learning_rates.append(self.optimizer.param_groups[0]['lr'])

            # Using our chosen device
            targets = batch["targets"][:, self.target].to(self.device).unsqueeze(dim=-1)
            # Standardizing the data
            targets = (targets - self.mean[self.target])/self.std[self.target]  

            # Backpropagate using the selected loss
            outputs = self.model(batch)
            loss = self.loss(outputs, targets)
            swa_loss.append(loss.item())

            # Tracking the results of the epoch
            mean_loss = mean_loss + loss
            
            mean_mae = mean_mae + self.metric(outputs*self.std[self.target] + self.mean[self.target], 
                                                    targets*self.std[self.target] + self.mean[self.target])
            swa_metric.append(self.metric(outputs*self.std[self.target] + self.mean[self.target], 
                                                    targets*self.std[self.target] + self.mean[self.target]).item())
            # Tracking loss during training
            if batch_idx%100 == 0:
                print(f"Current loss {mean_loss.item()/(batch_idx+1)} Current batch {batch_idx}/{len(self.train_set)} ({100*batch_idx/len(self.train_set):.2f}%)")

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Cleanup at the end of the batch
            del batch
            del targets
            del loss
            del outputs
            torch.cuda.empty_cache()
        
            # SWA update if we end a cycle
            if batch_idx % c == 0 and batch_idx != 0:
                n_models = batch_idx / c
                current_weights = self.model.get_weights()
                # Average the weights and update the model
                for swa_layer, layer in zip(swa_weights, current_weights):
                    swa_layer = (swa_layer * n_models + layer) / n_models

        # Printing the result of the epoch 
        if self.target not in [0, 1, 5, 11, 16, 17, 18]:
            mean_mae = mean_mae * 1000
        print("[SWA] MAE for the training set", mean_mae.item()/(batch_idx + 1))
        print("Taking weight average of SWA as weights")
        self.model.update_weights(weights = swa_weights)
        self.plot_data_swa(swa_loss, swa_metric, swa_learning_rates, c)

        # Tracking results for plotting
        self.learning_curve.append(mean_mae.item()/(batch_idx + 1))
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)

    def _train_epoch_swa_acyclical(self, weights: list, step: int = 1, alpha: float = 0.005) -> dict:
        """ Training logic for an epoch with Stochastic Weight Averaging with one cycle per epoch 
        Args:
            weights: original weights to load at the beginning of each epoch
            step: number of SWA steps already done (to compute the running average)
            alpha: learning rate to use during the SWA process
        """
        mean_loss = torch.zeros(1).to(self.device)
        mean_mae = torch.zeros(1).to(self.device)

        # Store previous weights as swa weights
        swa_weights = self.model.get_weights()
        # Reinitialize the network to initial weights
        self.model.update_weights(weights = weights)
        # Modify current learning rate
        self.optimizer.param_groups[0]['lr'] = alpha

        for batch_idx, batch in enumerate(self.train_set):

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
                print(f"Current loss {mean_loss.item()/(batch_idx+1)} Current batch {batch_idx}/{len(self.train_set)} ({100*batch_idx/len(self.train_set):.2f}%)")

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Cleanup at the end of the batch
            del batch
            del targets
            del loss
            del outputs
            torch.cuda.empty_cache()
        
        # SWA update if we end an epoch
        current_weights = self.model.get_weights()
        # Average the weights and update the model
        for swa_layer, layer in zip(swa_weights, current_weights):
            swa_layer = (swa_layer * step + layer) / step

        # Printing the result of the epoch 
        if self.target not in [0, 1, 5, 11, 16, 17, 18]:
            mean_mae = mean_mae * 1000
        print("[SWA] MAE for the training set", mean_mae.item()/(batch_idx + 1))
        print("Taking weight average of SWA as weights")
        self.model.update_weights(weights = swa_weights)

        # Tracking results for plotting
        self.learning_curve.append(mean_mae.item()/(batch_idx + 1))
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)

    def _eval_model(self):
        """ Evaluating the current model on tbe validation data
        """
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

    def standardize_data(self):
        """ Calculate means and standard deviations
        """
        train_set = []
        for _, batch in enumerate(self.train_set):
            train_set.append(batch['targets'])
        train_set = torch.cat(train_set).squeeze(dim=1)

        return torch.mean(train_set, axis=-2), torch.std(train_set, axis=-2)

    def plot_data(self):
        """ Plotting the data from the training process
        """
        p_data = (self.learning_curve, self.valid_curve, self.learning_rates)
        plot_names = ['Training metric','Validation metric', 'Learning rate']
        x_names = ['MAE', 'MAE', 'LR']
        for i in range(3):
            self.summaries_axes[i].plot(range(len(p_data[i])), p_data[i])
            self.summaries_axes[i].set_ylabel(x_names[i])
            self.summaries_axes[i].set_xlabel('Epochs')
            self.summaries_axes[i].set_xlim((0, len(p_data[i])))
            self.summaries_axes[i].set_title(plot_names[i])

        plt.tight_layout()
        plt.savefig('Loss_plot.png', dpi=800)
        plt.close()

    def plot_data_swa(self, loss, metric, lr, c):
        """ Plotting the data from a SWA (cyclical) process
        Args:
            loss: loss during multiple cycles of the SWA process
            metric: metric during multiple cycles of the SWA process
            lr: lr during multiple cycles of the SWA process
            c: number of cycles to plot
        """
        swa_fig, swa_axs = plt.subplots(1,3, figsize=(10,5))
        swa_data = (loss, metric, lr)
        plot_names = ['Loss SWA','Metric SWA', 'LR SWA']
        x_names = ['MSE', 'MAE', 'LR']
        for i in range(3):
            swa_axs[i].plot(range(len(swa_data[i][:5*c])), swa_data[i][:5*c])
            swa_axs[i].set_ylabel(x_names[i])
            swa_axs[i].set_xlabel('Batches')
            swa_axs[i].set_xlim((0, len(swa_data[i][:5*c])))
            swa_axs[i].set_title(plot_names[i])

        plt.tight_layout()
        plt.savefig('SWA_plot.png', dpi=800)
        plt.close()


        