import torch

from PaiNN.data_loader import PaiNNDataLoader
from PaiNN.model import PaiNNModel
from PaiNN.trainer import Trainer
from PaiNN.utils import mse

def training():
        train_set = PaiNNDataLoader(r_cut=2, batch_size=32)
        model = PaiNNModel(r_cut=2)
        optimizer = torch.optim.Adam(params=model.parameters(), lr = 1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
        trainer = Trainer(
            model=model,
            loss=mse,
            target=0,
            optimizer=optimizer,
            data_loader=train_set,
            scheduler=scheduler,
            device="cpu"
        )
        trainer._train_epoch() 

if __name__=="__main__":
    training()