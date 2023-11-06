import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import QM9
from tqdm import tqdm
import math

class PaiNNDataset(Dataset):
    """ Class for dataset from QM9 data folder """

    def __init__(self, path: str = "../data", r_cut: float = 1., self_edge: bool = False):
        """ Constructor
        Args:
            path: file path for the dataset
        """
        self.data = QM9(root = path)[:10]
        self.r_cut = r_cut
        self.self_edge = self_edge

    def add_edges(self, pos) -> torch.Tensor:
        """ Return the edges between the atoms based on r_cut (adjacency matrix) """
        n_atoms = pos.shape[0]
        # Finding the number of edge
        edges = torch.zeros(n_atoms, n_atoms)
        # We set the diagonal to 1 if we want self.connection
        if self.self_edge:
            edges = torch.eyes(n_atoms, n_atoms)    
        
        # Calculating d_{i,j} for atom i and j
        for i in range(n_atoms):
            for j in range(i):
                if torch.linalg.norm(pos[i] - pos[j]) <= self.r_cut:
                    edges[i][j], edges[j][i] = 1, 1

        return edges

    def __len__(self):
        """ Return the length of the dataset """
        return len(self.data)

    def __getitem__(self, idx) -> torch.Tensor:
        """ Return the sample corresponding to idx """
        # Add the adjacency matrix
        edges = self.add_edges(self.data[idx]['pos'])
        mol = self.data[idx].clone().detach()

        # The last N columns (where N is the number of columns) will be the adjacency matrix    
        return {'Z': mol['z'], 'pos': mol['pos'], 'edges': edges, 'targets': mol['y'], 'id': torch.tensor([mol['idx'].item()] * edges.shape[0])}
