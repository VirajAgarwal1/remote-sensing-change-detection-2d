from model import BaseNet
import torch
from dataloader import val_dataset, tensor_to_image
import numpy as np
import matplotlib.pyplot as plt


model = BaseNet(3, 1)
checkpoint = torch.load("./best_model.pth", map_location=torch.device("cpu"))
# model.load_state_dict(checkpoint['state_dict'])
model.load_state_dict(checkpoint, strict=False)


numPreds = 3

l = np.random.randint(0, len(val_dataset) - numPreds - 1)

fig, axes = plt.subplots(nrows=numPreds, ncols=7, figsize=(20, 10))
fig.tight_layout()

for i in range(numPreds):
    sample = val_dataset[l]
    o2, o3, o4, o5 = model(
        sample["img_A"].clone().reshape((1, 3, 256, 256)),
        sample["img_B"].clone().reshape((1, 3, 256, 256)),
    )
    l += 1
    plt.subplot(numPreds, 7, 7 * (i) + 1)
    plt.imshow(tensor_to_image(sample["img_A"]))

    plt.subplot(numPreds, 7, 7 * (i) + 2)
    plt.imshow(tensor_to_image(sample["img_B"]))

    plt.subplot(numPreds, 7, 7 * (i) + 3)
    plt.imshow(tensor_to_image(sample["label"]), cmap="gray")

    plt.subplot(numPreds, 7, 7 * (i) + 4)
    plt.imshow(tensor_to_image(o2.reshape((1, 256, 256))), cmap="gray")

    plt.subplot(numPreds, 7, 7 * (i) + 5)
    plt.imshow(tensor_to_image(o3.reshape((1, 256, 256))), cmap="gray")

    plt.subplot(numPreds, 7, 7 * (i) + 6)
    plt.imshow(tensor_to_image(o4.reshape((1, 256, 256))), cmap="gray")

    plt.subplot(numPreds, 7, 7 * (i) + 7)
    plt.imshow(tensor_to_image(o5.reshape((1, 256, 256))), cmap="gray")

plt.savefig("foo.png")
