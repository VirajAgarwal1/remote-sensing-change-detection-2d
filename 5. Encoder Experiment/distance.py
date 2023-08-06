"""
This package will determine scalar distance between 2 high dimensional vectors.
"""
import torch

def scalar_dist (enc1 :torch.Tensor, enc2 :torch.Tensor):
    """
    Input:
        enc1 -> torch.Tensor
        enc2 -> torch.Tensor
    Output:
        dist -> float
    """
    diff = torch.sqrt(torch.sum(torch.square(enc1 - enc2))).detach().numpy() # l2-norm
    return diff