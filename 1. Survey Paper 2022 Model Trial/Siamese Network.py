import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchsummary as summary


IMG_SIZE = 256
IMG_CHANNELS = 3

# # Gettiig the Dataset
# !wget https://github.com/maticvl/dataHacker/raw/master/DATA/at%26t.zip
# !rm -rf data
# !unzip "at&t.zip" -d


class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        # We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # Look untill the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # Look untill a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (
            img0,
            img1,
            torch.from_numpy(
                np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)
            ),
        )

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


# # Load the training dataset
# folder_dataset = datasets.ImageFolder(root="./data/faces/training/")
# # Resize the images and transform to tensors
# transformation = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor() ])
# # Initialize the network
# siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset, transform=transformation)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Convolutional Neural Network Layer
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        # Setting up the Fully Connected Layer
        self.fcn = nn.Sequential(
            nn.Linear(in_features=384, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=2),
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fcn(output)
        return output

    def forward(self, img1, img2):
        out1 = self.forward_once(img1)
        out2 = self.forward_once(img2)

        if out1 == out2:
            print("The outputs of bothe networks have computed out to be same...")

        return out1, out2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        euclidean_dist = F.pairwise_distance(out1, out2, keepdim=True)

        loss_euclidean = torch.mean(
            (1 - label) * torch.pow(euclidean_dist, 2)
            + (label) * torch.pow(torch.clamp(self.margin - euclidean_dist, min=0.0), 2)
        )

        return loss_euclidean


def models_same_weights(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


class CNN_select(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CNN_select, self).__init__()

        if kernel_size == 3:
            self.cnn = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        else:
            self.cnn = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.cnn(x)
        out = self.bn(out)
        out = self.act(out)

        return out


def CNN_Block(cnn1, cnn2, x):
    out = cnn1(x)
    out = cnn2(out)
    return out


class MultiScale_SNet_ShareWt(nn.Module):
    def __init__(self):
        super(MultiScale_SNet_ShareWt, self).__init__()

        self.cnn_3 = nn.ModuleList([])
        in_out_channels_3 = [(IMG_CHANNELS, 8), (8, 8), (16, 16), (32, 32), (64, 64)]
        for i in range(5):
            self.cnn_3.append(
                CNN_select(in_out_channels_3[i][0], in_out_channels_3[i][1], 3)
            )

        self.cnn_2 = nn.ModuleList([])
        in_out_channels_2 = [(8, 16), (16, 32), (32, 64), (64, 128)]
        for i in range(4):
            self.cnn_2.append(
                CNN_select(in_out_channels_2[i][0], in_out_channels_2[i][1], 2)
            )

        self.convT = nn.ConvTranspose2d(128, IMG_CHANNELS, 16, 16)

        self.alpha = nn.Parameter(
            torch.tensor(np.random.random_sample(), requires_grad=True).float()
        )
        self.beta = nn.Parameter(
            torch.tensor(np.random.random_sample(), requires_grad=True).float()
        )
        self.gamma = nn.Parameter(
            torch.tensor(np.random.random_sample(), requires_grad=True).float()
        )
        self.delta = nn.Parameter(
            torch.tensor(np.random.random_sample(), requires_grad=True).float()
        )

        self.max_pool_3d_8 = nn.MaxPool3d((8, 1, 1), (8, 1, 1), 0)
        self.max_pool_3d_4 = nn.MaxPool3d((4, 1, 1), (4, 1, 1), 0)
        self.max_pool_3d_2 = nn.MaxPool3d((2, 1, 1), (2, 1, 1), 0)

        self.mul = torch.multiply

    def forward_once(self, img):
        out = self.cnn_3[0](img)

        out = CNN_Block(self.cnn_3[1], self.cnn_2[0], out)
        x1 = out.clone()
        x1 = self.max_pool_3d_8(x1)
        x1 = x1.reshape((-1, 128, 16, 16))
        x1 = self.mul(self.alpha, x1)

        out = CNN_Block(self.cnn_3[2], self.cnn_2[1], out)
        x2 = out.clone()
        x2 = self.max_pool_3d_4(x2)
        x2 = x2.reshape((-1, 128, 16, 16))
        x2 = self.mul(self.beta, x2)

        out = CNN_Block(self.cnn_3[3], self.cnn_2[2], out)
        x3 = out.clone()
        x3 = self.max_pool_3d_2(x3)
        x3 = x3.reshape((-1, 128, 16, 16))
        x3 = self.mul(self.gamma, x3)

        out = CNN_Block(self.cnn_3[4], self.cnn_2[3], out)
        out = self.mul(self.delta, out)

        return x1, x2, x3, out

    def forward_feature_extr(self, img1, img2):
        x1, x2, x3, x4 = self.forward_once(img1)
        x5, x6, x7, x8 = self.forward_once(img2)

        out = x4 + x8
        out += x3 + x7
        out += x2 + x6
        out += x1 + x5

        return out

    def forward(self, img1, img2):
        out = self.forward_feature_extr(img1, img2)
        out = self.convT(out)

        return out


model = MultiScale_SNet_ShareWt()
for param in model.parameters():
    param.requires_grad = True


class A(nn.Module):
    def __init__(self):
        super().__init__()

        self.fcn1 = nn.Linear(128, 512)
        self.act = nn.ReLU(inplace=True)
        self.fcn2 = nn.Linear(512, 1024)
        self.fcn3 = nn.Linear(1024, 512)
        self.fcn4 = nn.Linear(512, 128)

    def forward(self, x):
        x = self.fcn1(x)
        x = self.act(x)
        x = self.fcn2(x)
        x = self.act(x)
        x = self.fcn3(x)
        x = self.act(x)
        x = self.fcn4(x)
        return x


class B(nn.Module):
    def __init__(self):
        super().__init__()

        self.A = A()
        self.fcn1 = nn.Linear(128, 10)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.A(x)
        x = self.fcn1(x)
        x = self.sig(x)
        return x


model = B()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 10

# training routine
for epoch in range(num_epochs):
    for x, y in dataloader:
        y_pred = model(x)
        loss = F.cross_entropy(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
