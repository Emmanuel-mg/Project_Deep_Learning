import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import QM9

class PaiNNDataset(Dataset):
    """ Class for dataset from QM9 data folder """

    def __init__(self, r_cut: float, path: str, self_edge: bool = False):
        """ Constructor
        Args:
            r_cut: radius used to link atoms in the molecul
            path: file path for the dataset
            self_edge: wether or not we should connect atoms to themselves
        """
        self.data = QM9(root = path)
        self.r_cut = r_cut
        self.self_edge = self_edge

    def add_edges(self, pos: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """ Return the edges between the atoms based on r_cut (adjacency matrix) 
        Args:
            pos: 3D positions of the different atoms in the moleculs
        """
        n_atoms = pos.shape[0]

        # Finding each edge and adding the coordinates to the list
        edges_coord = []
        dist = []
        normalized = []
        for i in range(n_atoms):
            for j in range(i + 1):
                if i==j and self.self_edge:
                    edges_coord.append([i,j])

                diff = pos[j] - pos[i]  
                dist_ij = torch.linalg.norm(diff)
                if dist_ij <= self.r_cut and i!=j:
                    edges_coord.append([i,j])
                    edges_coord.append([j,i])
                    dist.append(dist_ij.item())
                    dist.append(dist_ij.item())    # Same distance ij or ji
                    normalized.append((diff/dist_ij).tolist())
                    normalized.append((-diff/dist_ij).tolist())

        return torch.tensor(edges_coord), torch.tensor(dist).unsqueeze(dim=-1), torch.tensor(normalized)

    def __len__(self):
        """ Return the length of the dataset """
        return len(self.data)

    def __getitem__(self, idx) -> torch.Tensor:
        """ Return the sample corresponding to idx
        Args:
            idx: idx of the item to return from the dataset
        """
        # Add the adjacency matrix
        edges_coord, dist, normalized = self.add_edges(self.data[idx]['pos'])
        mol = self.data[idx].clone().detach()

        # The last N columns (where N is the number of columns) will be the adjacency matrix    
        return {'z': mol['z'], 'pos': mol['pos'], 'coord_edges': edges_coord, 'edges_dist': dist, 'normalized': normalized, 'targets': mol['y'], 'n_atom':  mol['z'].shape[0]}
