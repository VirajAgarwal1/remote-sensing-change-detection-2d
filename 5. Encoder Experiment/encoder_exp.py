"""
This package will do the actual encoder experiment.
"""
import torch
from encoder import Encoder
from distance import scalar_dist
import pickle
from dataloader import val_dataset


# write list to binary file
def write_list(a_list, filename):
    # store list in binary file so 'wb' mode
    with open(filename, 'wb') as fp:
        pickle.dump(a_list, fp)
        print('Done writing list into a binary file')

# Read list to memory
def read_list(filename):
    # for reading also binary mode is important
    with open(filename, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list


def split_image_label (imgA :torch.Tensor, label :torch.Tensor):
    """
    Input: 
        imgA  -> torch.Tensor , 3x256x256\n
        label -> torch.Tensor , 1x256x256
    Output:
        img_1 -> split image where change=1 in label , 3x256x256
        img_2 -> split image where change=0 in label , 3x256x256
    """
    img_1 = imgA.detach().clone()
    img_2 = imgA.detach().clone()

    img_1[0] = torch.where(label[0] == 1, img_1[0], 0.0)
    img_1[1] = torch.where(label[0] == 1, img_1[1], 0.0)
    img_1[2] = torch.where(label[0] == 1, img_1[2], 0.0)

    img_2[0] = torch.where(label[0] != 1, img_2[0], 0.0)
    img_2[1] = torch.where(label[0] != 1, img_2[1], 0.0)
    img_2[2] = torch.where(label[0] != 1, img_2[2], 0.0)

    return img_1, img_2


# MODEL BEING EXPERIMENTED WITH
MODEL_NAME  = "<model_name>"

# Data to be collected during the loop
green_data = []     # Green means that there was no change between the 2 images
num_green_data = 0
blue_data = []      # Blue means that images were striped of the part that had the change and then inferenced
num_blue_data = 0
red_data = []       # Red means that there was change between the 2 images
num_red_data = 0
yellow_data = []    # Yellow means that images were striped of the part which had no change and then inferenced
num_yellow_data = 0

# Loop over all the datapoints in the validation dataset
for i in range(len(val_dataset)):
    sample = val_dataset[i]
    imgA = sample["img_A"]
    imgB = sample["img_B"]
    label = sample["label"]

    # Encodings for green and red data
    model_enc = Encoder(MODEL_NAME)
    enc1, enc2 = model_enc(imgA, imgB)
    if label.max() > 0:
        # Red Data
        num_red_data += 1
        red_data.append(scalar_dist(enc1, enc2))
    else:
        # Green Data
        num_green_data += 1
        green_data.append(scalar_dist(enc1, enc2))


    # Encodings for blue and yellow data
    imgA_1, imgA_2 = split_image_label(imgA, label)
    imgB_1, imgB_2 = split_image_label(imgB, label)

    enc1_2 , enc2_2 = model_enc(imgA_1, imgB_1) # Encodings for change part of images -> Yellow Data encodings
    enc1_3 , enc2_3 = model_enc(imgA_2, imgB_2) # Encodings for no-change part of images -> Blue Data encodings

    # Blue Data
    num_blue_data += 1
    blue_data.append(scalar_dist(enc1_3, enc2_3))

    # Yellow Data
    num_yellow_data += 1
    yellow_data.append(scalar_dist(enc1_2, enc2_2))

write_list(green_data, MODEL_NAME+"-green_data")
write_list(blue_data, MODEL_NAME+"-blue_data")
write_list(red_data, MODEL_NAME+"-red_data")
write_list(yellow_data, MODEL_NAME+"-yellow_data")
