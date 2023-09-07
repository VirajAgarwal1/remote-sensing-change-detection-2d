"""
This package is responsible for getting the encoder in required format. The encoder returned by this package 
of specified model will have common qualities like common way to pass input into the encoder and output is 
2 encodings of the respective inputs. 
This makes coding procedure more straight-forward for other parts of the program.
"""
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rdpnet_model_enc():
    pass


def fccdn_model_enc():
    from FCCDN_network.GenerateNet import GenerateNet

    class Config:
        def __init__(self):
            self.MODEL_NAME = "FCCDN"
            self.MODEL_OUTPUT_STRIDE = 16
            self.BAND_NUM = 3
            self.USE_SE = True

    cfg = Config()
    model = GenerateNet(cfg)
    checkpoint = torch.load(
        "./FCCDN_network/FCCDN_test_LEVIR_CD.pth", map_location=torch.device("cpu")
    )
    model.load_state_dict(checkpoint, strict=False)

    return model


def a2net_model_enc():
    from A2_network.model import BaseNet

    model = BaseNet(3, 1)
    checkpoint = torch.load(
        "./A2_network/best_model.pth", map_location=torch.device("cpu")
    )
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint, strict=False)
    return model


class Encoder(nn.Module):
    """
    Input:
        imgA -> torch.Tensor , 3x256x256 , No augmentation
        imgB -> torch.Tensor , 3x256x256 , No augmentation
    Output:
        enc1 -> Encoding for imgA , torch.Tensor
        enc2 -> Encoding for imgB , torch.Tensor
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        if model_name.lower() == "rdpnet":
            self.model = rdpnet_model_enc()
        if model_name.lower() == "fccdn":
            self.model = fccdn_model_enc()
        if model_name.lower() == "a2net":
            self.model = a2net_model_enc()

    def forward(self, imgA: torch.Tensor, imgB: torch.Tensor):
        enc1, enc2 = self.model(imgA, imgB)
        enc1 = torch.flatten(enc1)
        enc2 = torch.flatten(enc2)

        return enc1, enc2
