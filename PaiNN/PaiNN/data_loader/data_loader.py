import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from PaiNN.dataset import PaiNNDataset

class PaiNNDataLoader(DataLoader):
    """ PaiNNDataLoader to load PaiNN training data """

    def __init__(self, data_path: str = "../data", batch_size: int = 32, r_cut: float = 2., self_edge: bool = False, test_split: float = 0.1, validation_split: float = 0.2, nworkers: int = 2):
        """ Constructor
        Args:
            train_path: path to the training dataset
            test_path: path to the test dataset(s)
            batch_size: size of the batch
            shuffle: shuffles the data 
            test_split: decimal for the split of the test (on the entire dataset)
            validation_split: decimal for the split of the validation (on the training dataset)
            nworkers: workers for the dataloader class
        """    

        self.dataset = PaiNNDataset(path = data_path, r_cut = r_cut, self_edge = self_edge)
        self.length = len(self.dataset)
        self.train_sampler = SubsetRandomSampler(np.array(range(self.length)))
        self.valid_sampler = None
        self.test_sampler = None

        if test_split:
            self.test_sampler = self._split(test_split)

        if validation_split:
            self.valid_sampler = self._split(validation_split)

        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': nworkers
        }

        # Return the training dataset
        super().__init__(self.dataset, sampler=self.train_sampler, collate_fn=self.collate_fn, **self.init_kwargs)

    # We need to define our custom collate_fn because our samples (molecule) have different size
    # ie. you cannot use torch.stack on it
    def collate_fn(self, data):
        # Each mol is a dic with "z" = n_atoms here we get a dic with "z" = (n_mol, n_atoms_mol)
        batch_dict = {k: [dic[k] for dic in data] for k in data[0].keys()} 

        # We need to define the id and the edges_coord differently (because we begin indexing from 0)
        id = []
        edges_coord = []
        delta = 0
        for i, mol in enumerate(data):
            # Unique ID per batch (beginning from 0)
            id += [i] * mol["n_atom"]
            # Adding the size of the molecule before in order to keep the indexes true in the final A matrix
            edges_coord += [[x + delta, y + delta] for x,y in  mol["coord_edges"]]
            delta = mol["n_atom"]

        return {'z': torch.cat(batch_dict['z']), 'pos': torch.cat(batch_dict['pos']), 'graph': torch.tensor(edges_coord), 'graph_idx': torch.tensor(id), 'targets': torch.cat(batch_dict['targets'])}

    def _split(self, validation_split: float):
        """ Creates a sampler to extract training and validation data
        Args:
            validation_split: decimal for the split of the validation
        """    
        train_idx = np.array(range(self.length))

        # Getting randomly the index of the validation split (we therefore don't need to shuffle)
        split_idx = np.random.choice(
            train_idx, 
            int(self.length*validation_split), 
            replace=False
        )
        
        # Deleting the corresponding index in the training set
        train_idx = np.delete(train_idx, split_idx)

        # Getting the corresponding PyTorch samplers
        train_sampler = SubsetRandomSampler(train_idx)
        self.train_sampler = train_sampler

        return SubsetRandomSampler(split_idx)

    def get_val(self) -> list:
        """ Return the validation data"""
        if self.valid_sampler is None:
            return None
        else: 
            return DataLoader(self.dataset, sampler=self.valid_sampler, collate_fn=self.collate_fn, **self.init_kwargs)

    def get_test(self) -> list:
        """ Return the test data"""
        if self.test_sampler is None:
            return None
        else: 
            return DataLoader(self.dataset, sampler=self.test_sampler, collate_fn=self.collate_fn, **self.init_kwargs)
