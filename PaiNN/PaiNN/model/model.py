import torch
import torch.nn as nn
import numpy

from PaiNN.data_loader import PaiNNDataLoader
from PaiNN.utils import rbf, cos_cut

class PaiNNModel(nn.Module):
    """ PaiNN model architecture """

    def __init__(self,  r_cut: float, hidden_state_size: int = 128):
        """ Constructor
        Args:
            hidden_state_size: size of the embedding features
        """
        # Instantiate as a module of PyTorch
        super(PaiNNModel, self).__init__()

        # Parameters of the model
        self.r_cut = r_cut


        # Embedding layer for our model
        num_embedding = 119 # number of all elements in the periodic table
        self.hidden_state_size = hidden_state_size
        self.embedding_layer = nn.Embedding(num_embedding, self.hidden_state_size)

    def forward(self, input):
        """ Forward pass logic 
        Args:
            input: dictionnary coming from data_loader
        """
        graph = input['graph']
        edges_dist = input['edges_dist']
        edges_sense = input['orientation']

        # Outputs from the atomic numbers
        node_scalars = self.embedding_layer(input['z'])

        # Initializing the node vector
        node_vectors = torch.zeros((graph.shape[0], 3, self.hidden_state_size), 
                                  device = edges_dist.device, 
                                  type = edges_dist.type
                                  )

        predict = {}

        return predict
    

class Message(nn.Module):
    """ Message block from PaiNN paper"""
    def __init__(self, node_size: int, rbf_size: int, r_cut: float):
        """ Constructor
        Args:
            node_size: size to use in the atomwise layers (node_size to 3*node_size)
            rbf_size: number of radial basis functions to use in RBF
            r_cut: radius to cutoff interaction
        """
        super.__init__()
        # Atomwise layers applied to node scalars
        self.atomwise_layers = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, 3 * node_size)
        )
        
        # RBF and cosine cutoff parameters
        self.rbf_dim = rbf_size
        self.r_cut = r_cut
        # rotationally-invariant filters
        self.expand_layer = nn.Linear(self.rbf_dim, 384)

    def forward(self, node_scalars: torch.Tensor, node_vectors: torch.Tensor, graph: torch.Tensor, edges_dist: torch.Tensor, edges_sense: torch.Tensor):
        """ Forward pass
        Args:
            node_scalars: scalar representations of the atoms 
            node_vectors: vector (equivariant) representations of the atoms
            graph: interactions between atoms (base on r_cut)
            edges_dist: distances between neighbours
            r_cut: radius to cutoff interaction
        """
        # Outputs from scalar representations 
        atomwise_rep = self.atomwise_layers(node_scalars)

        # Outputs from edges distances
        filter_rbf = rbf(edges_dist, 
                          r_cut = self.r_cut,
                          rbf = self.rbf_dim
                          )
        filter_out = self.expand_layer(filter_rbf)
        cosine_cutoff = cos_cut(edges_dist,
                                r_cut = self.r_cut
                                )
        dist_rep = filter_out * cosine_cutoff

        # Getting the Hadamard product by selecting the neighbouring atoms
        residual = atomwise_rep[graph[:,1]] * dist_rep

        # Splitting the output
        residual_vectors, residual_scalars, direction_rep = residual.split(128, dim=-1)

        # Hadamard product with the neighbours vectors representation
        residual_vectors = node_vectors[graph[:, 1]] * residual_vectors
        # Hadamard product between the direction representations and the sense of the edges
        residual_directions = edges_sense * direction_rep
        residual_vectors += residual_directions

        node_scalars += torch.zeros_like(node_scalars).index_add_(0, graph[:, 0], residual_scalars)
        node_vectors += torch.zeros_like(node_vectors).index_add_(0, graph[:, 0], residual_vectors)

        return node_scalars, node_vectors
    
class Update(nn.Module):
    """ Message block from PaiNN paper"""
    def __init__(self, node_size: int):
        """ Constructor
        Args:
            node_size: size to use in the atomwise layers (node_size to 3*node_size)
            rbf_size: number of radial basis functions to use in RBF
            r_cut: radius to cutoff interaction
        """
        super.__init__()
        self.node_size = node_size

        # U and V matrices 
        self.U = nn.Linear(node_size, node_size, bias = False)
        self.V = nn.Linear(node_size, node_size, bias = False)
        
        # Atomwise layers applied to node scalars and V projections (stacked)
        self.atomwise_layers = nn.Sequential(
            nn.Linear(2 * node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, 3 * node_size)
        )


    def forward(self, node_scalars: torch.Tensor, node_vectors: torch.Tensor):
        """ Forward pass
        Args:
            node_scalars: scalar representations of the atoms 
            node_vectors: vector (equivariant) representations of the atoms
            graph: interactions between atoms (base on r_cut)
            edges_dist: distances between neighbours
            r_cut: radius to cutoff interaction
        """
        # Outputs from matrix projection
        Uv = self.U(node_vectors)
        Vv = self.V(node_vectors)

        # Stacking V projections and node scalars
        node_scalars_Vv = torch.stack((node_scalars, torch.linalg.norm(Vv, dim=1)), dim=1)
        a = self.atomwise_layers(node_scalars_Vv)
        avv, asv, ass = a.split(self.node_size, dim=-1)

        # Scalar product between Uv and Vv
        scalar_product = torch.sum(Uv * Vv, dim=-1)

        # Calculating the residual values for scalars and vectors
        residual_scalars = ass + asv * scalar_product
        residual_vectors = avv * (Uv.unsqueeze(dim=1)).expand(-1, 3, -1)

        # Updating the representations
        node_scalars += residual_scalars
        node_vectors += residual_vectors

        return node_scalars, node_vectors
    
if __name__=="__main__":
    train_set = PaiNNDataLoader(batch_size=2)
    model = PaiNNModel()
    val_set = train_set.get_val()
    test_set = train_set.get_test()
    for i, batch in enumerate(train_set):
        model(batch)