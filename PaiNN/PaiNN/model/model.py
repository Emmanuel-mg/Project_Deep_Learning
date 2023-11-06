import torch
import torch.nn as nn


class PaiNNModel(nn.Module):
    def __init__(self):
        embedding_layer = nn.Embedding()

    def forward(self, *input):
        """ Forward pass logic """

        raise NotImplementedError