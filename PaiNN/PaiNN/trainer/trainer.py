import torch
import numpy as np

from PaiNN.data_loader import PaiNNDataLoader
from PaiNN.model import PaiNNModel

train_set = PaiNNDataLoader(batch_size=2)
model = PaiNNModel()
val_set = train_set.get_val()
test_set = train_set.get_test()

if __name__=="__main__":
    for i, batch in enumerate(train_set):
        print(batch.keys())
        model(batch)