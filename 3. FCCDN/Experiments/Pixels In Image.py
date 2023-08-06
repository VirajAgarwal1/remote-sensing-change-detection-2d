from networks import GenerateNet
from loss import FCCDN_loss_BCD
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from dataloader import val_dataset, tensor_to_image
# from receptivefield.pytorch import PytorchReceptiveField
# from receptivefield.image import get_default_image

class Config():
    def __init__(self):
        self.MODEL_NAME = 'FCCDN'
        self.MODEL_OUTPUT_STRIDE = 16
        self.BAND_NUM = 3
        self.USE_SE = True


cfg = Config()
model = GenerateNet(cfg)
checkpoint = torch.load("./training/FCCDN_plain/best_model/best_model.pt", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])



output = model(input)
loss = FCCDN_loss_BCD(output, label)
print("FCCDN_loss_BCD = {}".format(loss.item()))

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10,5))
fig.tight_layout()

for i in range(2):
    plt.subplot(2, 4, 4*(i) + 1)
    plt.imshow(tensor_to_image(data1[i].detach().clone()))

    plt.subplot(2, 4, 4*(i) + 2)
    plt.imshow(tensor_to_image(data2[i].detach().clone()))

    plt.subplot(2, 4, 4*(i) + 3)
    plt.imshow(tensor_to_image(label1[i].detach().clone()), cmap="gray")    

    plt.subplot(2, 4, 4*(i) + 4)
    plt.imshow(tensor_to_image(output[0][i].detach().clone()), cmap="gray")  

plt.savefig('foo.png')
