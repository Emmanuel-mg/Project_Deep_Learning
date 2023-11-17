import torch

from PaiNN.data_loader import PaiNNDataLoader
from PaiNN.model import PaiNNModel
from PaiNN.trainer import Trainer
from PaiNN.utils import mse

def training():
        train_set = PaiNNDataLoader(r_cut=5, batch_size=50)
        model = PaiNNModel(r_cut=5, device="cpu")
        optimizer = torch.optim.Adam(params=model.parameters(), lr = 1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
        trainer = Trainer(
            model=model,
            loss=mse,
            target=2,
            optimizer=optimizer,
            data_loader=train_set,
            scheduler=scheduler
        )
        trainer._train_epoch() 
        trainer.plot_data()

if __name__=="__main__":
    training()