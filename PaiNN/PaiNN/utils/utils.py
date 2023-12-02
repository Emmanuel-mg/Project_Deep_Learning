import torch

def rbf(inputs: torch.Tensor, r_cut: float, output_size: int = 20):
    """ Function
    Args:
        inputs: input to which we apply the rbf (usually it will be distances)
        r_cut: the radius at which we cut off
    """
    
    # We will apply it between 1 and output size (usually 1 and 20)
    n = torch.arange(1, output_size + 1).to(inputs.device)

    return torch.sin(n * torch.pi * inputs) / (r_cut * inputs)

def cos_cut(inputs: torch.Tensor, r_cut: float):
    """ Function
    Args:
        inputs: inputs on which we will apply Behler-style cosine cutoff
    """

    # We return the cosine cutoff for inputs smaller than the radius cutoff
    return 0.5 * (1 + torch.cos(torch.pi * inputs / r_cut)) * (inputs < r_cut).float()

def mse(preds: torch.Tensor, targets: torch.Tensor):
    return torch.mean((preds - targets).square())

def mae(preds: torch.Tensor, targets: torch.Tensor):
    return torch.mean(torch.abs(preds - targets))