import torch
import numpy as np
import matplotlib.pyplot as plt
from RDPNet import RDPNet
import random
import torchvision
from torch.utils.data import DataLoader
from dataloader import val_dataset, tensor_to_image

model = RDPNet(3, 2)
checkpoint = torch.load("./RDPNet_CDD.pth", map_location=torch.device('cpu'))
# print(checkpoint.keys())
model.load_state_dict(checkpoint)


ins = []
outs = []
labels = []
numPlots = 3

for i in range(numPlots):
    input1 = np.zeros((3,256,256))
    index1 = random.randint(4,250)
    index2 = random.randint(4,250)
    for j in range(3):
        input1[j][index1][index2] = 1
        input1[j][index1+1][index2] = 1
        input1[j][index1][index2+1] = 1
        input1[j][index1-1][index2] = 1
        input1[j][index1][index2-1] = 1
        input1[j][index1+1][index2+1] = 1
        input1[j][index1-1][index2-1] = 1
        input1[j][index1+1][index2-1] = 1
        input1[j][index1-1][index2+1] = 1
    input1 = input1.astype("float32")
    input1 = torch.Tensor(input1)

    input2 = torch.Tensor(np.zeros((3,256,256)).astype("float32"))

    input = torch.cat([input1.reshape((1,3,256,256)), input2.reshape((1,3,256,256))], dim=0)

    # sample = val_dataset[random.randint(0,len(val_dataset)-1)]
    # data1 = sample["img_A"].clone()
    # data2 = sample["img_B"].clone()
    # label = sample["label"].clone()

    output = model(input1.reshape((1,3,256,256)), input2.reshape((1,3,256,256)))
    # output = model(data1.reshape((1,3,256,256)), data2.reshape((1,3,256,256)))
    output = output.detach().numpy()
    # print(output.shape)
    out_fix = output[0][0].copy()
    for i in range(256):
        for j in range(256):
            out_fix[i][j] = np.argmax( [output[0][0][i][j], output[0][1][i][j]] )
    out_fix = out_fix.reshape(256,256,1)

    ins.append(input)
    # ins.append((data1.detach().clone().moveaxis(0,-1).numpy(), data2.detach().clone().moveaxis(0,-1).numpy()))
    # labels.append(tensor_to_image(label.detach().clone()))
    outs.append(out_fix)










fig, axes = plt.subplots(nrows=numPlots, ncols=3, figsize=(10,5))
fig.tight_layout()


for i in range(numPlots):
    plt.subplot(numPlots, 3, 3*(i) + 1)
    plt.imshow(ins[i][0].moveaxis(0,-1).numpy())

    plt.subplot(numPlots, 3, 3*(i) + 2)
    plt.imshow(ins[i][1].moveaxis(0,-1).numpy())

    plt.subplot(numPlots, 3, 3*(i) + 3)
    plt.imshow(outs[i], cmap="gray")

# for i in range(numPlots):
#     plt.subplot(numPlots, 4, 4*(i) + 1)
#     plt.imshow(ins[i][0])

#     plt.subplot(numPlots, 4, 4*(i) + 2)
#     plt.imshow(ins[i][1])

#     plt.subplot(numPlots, 4, 4*(i) + 3)
#     plt.imshow(labels[i])

#     plt.subplot(numPlots, 4, 4*(i) + 4)
#     plt.imshow(outs[i], cmap="gray")

plt.savefig('foo.png')