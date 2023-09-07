"""
This package will determine scalar distance between 2 high dimensional vectors.
"""
import torch
import numpy as np


def scalar_dist(enc1: torch.Tensor, enc2: torch.Tensor) -> torch.Tensor:
    """
    Input:
        enc1 -> torch.Tensor\n
        enc2 -> torch.Tensor
    Output:
        dist -> torch.Tensor[float32] , 1-dimensional
    """
    diff = torch.sqrt(torch.sum(torch.square(enc1 - enc2)))  # l2-norm
    return diff
