import torch

from PaiNN.data_loader import PaiNNDataLoader
from PaiNN.model import PaiNNModel
from PaiNN.trainer import Trainer
from PaiNN.utils import mse

def training():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{device} will be used for training the PaiNN model")
        train_set = PaiNNDataLoader(r_cut=5, 
                                    batch_size=50
        )
        model = PaiNNModel(r_cut=5, 
                           device=device
        )
        optimizer = torch.optim.Adam(params=model.parameters(), lr = 1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
        trainer = Trainer(
            model=model,
            loss=mse,
            target=2,
            optimizer=optimizer,
            data_loader=train_set,
            scheduler=scheduler,
            device=device
        )
        trainer._train(num_epoch = 20)
        trainer.plot_data()

if __name__=="__main__":
    training()