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

if __name__ == '__main__':

    final_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)
    
    data1,data2,input,label1,label2,label = [None]*6
    
    for i_batch, sample in enumerate(final_dataloader):
        data1 = sample["img_A"].clone()
        data2 = sample["img_B"].clone()
        input = [data1, data2]

        label1 = sample["label"].clone()
        label2 = torchvision.transforms.Resize(sample["label"].shape[2]//2, antialias=False)(label1)
        label = [label1.float(), label2.float()]
        break

    cfg = Config()
    model = GenerateNet(cfg)
    checkpoint = torch.load("./pretrained/FCCDN_test_LEVIR_CD.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=False)

    # # define model functions
    # def model_fn() -> nn.Module:
    #     model.eval()
    #     return model

    # input_shape = [96, 96, 3]
    # rf = PytorchReceptiveField(model_fn)
    # rf_params = rf.compute(input_shape = input_shape)

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
