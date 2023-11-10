import torch
import torch.nn as nn
import numpy

from PaiNN.data_loader import PaiNNDataLoader
from PaiNN.utils import rbf, cos_cut

class PaiNNModel(nn.Module):
    """ PaiNN model architecture """

    def __init__(self, hidden_state_size: int = 128):
        """ Constructor
        Args:
            hidden_state_size: size of the embedding features
        """
        # Instantiate as a module of PyTorch
        super(PaiNNModel, self).__init__()

        # Embedding layer for our model
        num_embedding = 119 # number of all elements in the periodic table
        self.embedding_layer = nn.Embedding(num_embedding, hidden_state_size)

        self.atomwise_layers = nn.Sequential(
            nn.Linear(hidden_state_size, 128),
            nn.SiLU(),
            nn.Linear(128, 384)
        )

    def forward(self, input):
        """ Forward pass logic 
        Args:
            input: dictionnary coming from data_loader
        """
        embedding = self.embedding_layer(input['z'])

        atomwise = self.atomwise_layers(embedding)

        return atomwise
    


if __name__=="__main__":
    train_set = PaiNNDataLoader(batch_size=2)
    model = PaiNNModel()
    val_set = train_set.get_val()
    test_set = train_set.get_test()
    for i, batch in enumerate(train_set):
        print(batch.keys())
        model(batch)