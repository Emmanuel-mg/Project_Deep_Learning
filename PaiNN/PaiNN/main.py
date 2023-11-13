import torch

from PaiNN.data_loader import PaiNNDataLoader
from PaiNN.model import PaiNNModel
from PaiNN.trainer import Trainer
from PaiNN.utils import mse

def training():
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