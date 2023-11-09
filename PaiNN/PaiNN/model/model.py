import torch
import torch.nn as nn
import numpy

from PROT.utils import rbf

class PaiNNModel(nn.Module):
    """ PaiNN model architecture """

    def __init__(self, hidden_state_size):
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
        raise NotImplementedError