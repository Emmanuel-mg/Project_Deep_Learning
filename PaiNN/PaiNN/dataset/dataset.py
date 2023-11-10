import torch
import math
from torch.utils.data import Dataset
from torch_geometric.datasets import QM9

class PaiNNDataset(Dataset):
    """ Class for dataset from QM9 data folder """

    def __init__(self, path: str = "../data", r_cut: float = 2., self_edge: bool = False):
        """ Constructor
        Args:
            path: file path for the dataset
        """
        self.data = QM9(root = path)
        self.r_cut = r_cut
        self.self_edge = self_edge

    def add_edges(self, pos) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """ Return the edges between the atoms based on r_cut (adjacency matrix) """
        n_atoms = pos.shape[0]

        # Finding each edge and adding the coordinates to the list
        edges_coord = []
        dist = []
        orientation = []
        for i in range(n_atoms):
            for j in range(i + 1):
                if i==j and self.self_edge:
                    edges_coord.append([i,j])

                diff = pos[j] - pos[i]  
                dist_ij = torch.linalg.norm(diff)
                if dist_ij <= self.r_cut and i!=j:
                    edges_coord.append([i,j])
                    edges_coord.append([j,i])
                    dist.append(dist_ij)
                    dist.append(dist_ij)    # Same distance ij or ji
                    orientation.append(diff/dist_ij)
                    orientation.append(-diff/dist_ij)
        return torch.tensor(edges_coord), torch.tensor(dist), torch.tensor(orientation)

    def __len__(self):
        """ Return the length of the dataset """
        return len(self.data)

    def __getitem__(self, idx) -> torch.Tensor:
        """ Return the sample corresponding to idx """
        # Add the adjacency matrix
        edges_coord, dist, orientation = self.add_edges(self.data[idx]['pos'])
        mol = self.data[idx].clone().detach()

        # The last N columns (where N is the number of columns) will be the adjacency matrix    
        return {'z': mol['z'], 'pos': mol['pos'], 'coord_edges': edges_coord, 'edges_dist': dist, 'orientation': orientation, 'targets': mol['y'], 'n_atom':  mol['z'].shape[0]}
