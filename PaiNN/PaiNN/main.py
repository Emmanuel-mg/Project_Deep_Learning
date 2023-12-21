import torch

from PaiNN.data_loader import PaiNNDataLoader
from PaiNN.model import PaiNNModel
from PaiNN.trainer import Trainer
from PaiNN.utils import mse, mae

def training(n_iterations, train_set):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{device} will be used for training the PaiNN model")
        model = PaiNNModel(r_cut = 5, 
                n_iterations = n_iterations,
                device = device
                ).to(device)

        optimizer = torch.optim.AdamW(params=model.parameters(), lr = 5e-4, weight_decay = 0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 5)
        trainer = Trainer(
            model = model,
            loss = mse,
            metric = mae,
            target = 2,
            optimizer = optimizer,
            data_loader = train_set,
            scheduler = scheduler,
            device = device
        )
        trainer._train(num_epoch = 3, epoch_swa = 1)
        trainer.plot_data()

if __name__=="__main__":
        train_set = PaiNNDataLoader(r_cut = 5, 
                                batch_size = 100
                )
        training(n_iterations = 3, train_set = train_set)