"""
This package is responsible for getting the encoder in required format. The encoder returned by this package 
of specified model will have common qualities like common way to pass input into the encoder and output is 
2 encodings of the respective inputs. 
This makes coding procedure more straight-forward for other parts of the program.
"""
import torch
import torch.nn as nn

def rdpnet_model_enc ():
    from RDPNet import 
    pass

class Encoder (nn.Module):
    def __init__ (self, model_name :str) -> None:
        super().__init__()
        if model_name.lower() == "rdpnet":
            self.model = rdpnet_model_enc()
            pass

    def forward (self, imgA :torch.Tensor, imgB :torch.Tensor):
        """
        Input:
            imgA -> torch.Tensor , 3x256x256 , No augmentation
            imgB -> torch.Tensor , 3x256x256 , No augmentation
        Output:
            enc1 -> Encoding for imgA , torch.Tensor
            enc2 -> Encoding for imgB , torch.Tensor
        """
        pass

