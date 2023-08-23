import os
from typing import Any
import torch
from skimage import io, util, exposure
from skimage import transform as ski_transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as torchvis_transforms
from torchvision import utils
from natsort import natsorted 
import PIL

THRESHOLD_HYP_PARAM = 0.7

class BuildingChangeDataset_Levir (Dataset):
    """LEVIR Building Change Dataset."""

    def __init__(self, root_dir_path :str, transforms=None) -> None:
        
        self.root_dir_path = root_dir_path
        self.transforms = transforms
        # Saving sorted image filenames so that they can be indexed appropriately
        # natosrted is used as normal sorted function is guving the desired result 
        # of "test_100_0.png" coming after "test_2_5.png"
        self.A_images = natsorted (os.listdir( os.path.join(self.root_dir_path,"A") ))
        self.B_images = natsorted (os.listdir( os.path.join(self.root_dir_path,"B") ))
        self.label_images = natsorted (os.listdir( os.path.join(self.root_dir_path,"label") ))

    def __len__ (self):
        return min( len(self.A_images), len(self.B_images), len(self.label_images) )
    
    def __getitem__(self, index: int) -> Any:
        """
        Input: 
            index: integer
        Ouput:
            sample: ( (A_image , B_image) , label )
                A_image: shape = (h, w, 3)
                B_image: shape = (h, w, 3)
                label: shape = (h, w)
        """
        
        img_A_name = self.A_images[index]
        img_A_path = os.path.join(self.root_dir_path, "A", img_A_name)
        # print(os.path.join(self.root_dir_path, "A", img_A_name))
        img_A = io.imread(img_A_path)

        img_B_name = self.B_images[index]
        img_B_path = os.path.join(self.root_dir_path, "B", img_B_name)
        # print(os.path.join(self.root_dir_path, "B", img_B_name))
        img_B = io.imread(img_B_path)
        
        img_label_name = self.label_images[index]
        img_label_path = os.path.join(self.root_dir_path, "label", img_label_name)
        # print(os.path.join(self.root_dir_path, "label", img_label_name))
        img_label = io.imread(img_label_path, as_gray=True)

        # The labels should only have integer values...
        img_label[ img_label >= THRESHOLD_HYP_PARAM ] = 1
        img_label[ img_label <  THRESHOLD_HYP_PARAM ] = 0

        sample = { # Returning the sample requested
            "img_A" : img_A,
            "img_B" : img_B,
            "label" : img_label
        }

        if self.transforms: # Making transformations on images if given
            sample = self.transforms(sample)

        return sample


class SplitImageLabel (object):

    def splitFunc (self, imgA :torch.Tensor, label :torch.Tensor) -> tuple[torch.Tensor]:
        """
        Input: 
            imgA  -> torch.Tensor , [3,256,256]\n
            label -> torch.Tensor , [1,256,256]
        Output:
            img_1 -> split image where change=1 in label , [3,256,256]\n
            img_2 -> split image where change=0 in label , [3,256,256]
        """
        img_1 = imgA.detach().clone()
        img_2 = imgA.detach().clone()

        label = label.expand((3,-1,-1)) # Repeating the mask for each channel 3 times since the same mask is applied to all the channels

        img_1 = torch.where(label == 1., imgA, 0.0)
        img_1 = torch.where(label == 1., imgA, 0.0)
        img_1 = torch.where(label == 1., imgA, 0.0)

        img_2 = torch.where(label != 1., imgA, 0.0)
        img_2 = torch.where(label != 1., imgA, 0.0)
        img_2 = torch.where(label != 1., imgA, 0.0)

        return img_1, img_2
    
    def __call__(self, sample) -> Any: 
        ##### Additional Steps needed to be taken in order to make the dataset for encoder model training
        """
        GREEN   ->  Original Images + No change                     Prob = 100  Expect = small  (0)
        BLUE    ->  Changed parts are striped + There are changes   Prob = 25   Expect = small  (0)
        RED     ->  Original Images + There are changes             Prob = 50   Expect = big    (1)
        YELLOW  ->  Not-changed part striped + There are changes    Prob = 25   Expect = big    (1)
        """

        img_A , img_B , label = sample["img_A"] , sample["img_B"] , sample["label"]

        # print("img_A type = ", type(img_A))
        # print("img_A shape = ", img_A.shape)
        # print("img_B type = ", type(img_B))
        # print("img_B shape = ", img_B.shape)
        # print("label type = ", type(label))
        # print("label shape = ", label.shape)

        ### Changing the images to ones we need...
        imgA_1, imgA_2 = self.splitFunc(img_A, label)
        imgB_1, imgB_2 = self.splitFunc(img_B, label)

        if label.max() > 0:
            # There is change
            p = torch.rand((1))[0]

            if p < 0.5:
                # Red Data
                label = torch.Tensor([1])

            elif p >= 0.5 and p < 0.75:
                # Blue Data
                img_A = imgA_2
                img_B = imgB_2
                label =  torch.Tensor([1])

            else:
                # Yellow Data
                img_A = imgA_1
                img_B = imgB_1
                label = torch.Tensor([1])

        else:
            # Green Data
            label =  torch.Tensor([1])

        output = { # Returning the sample requested
            "img_A" : img_A.requires_grad_(True),
            "img_B" : img_B.requires_grad_(True),
            "label" : label.requires_grad_(True)
        }

        return  output


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
    
        img_A , img_B , label = sample["img_A"] , sample["img_B"] , sample["label"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        img_A = np.moveaxis(img_A, -1, 0)
        img_B = np.moveaxis(img_B, -1, 0)
        # Making graysacle label image from (H,W) => (H,W,1) where 1 represents channels
        label = label.reshape((label.shape[0], label.shape[1], 1)) 
        label = np.moveaxis(label, -1, 0)

        img_A = torch.from_numpy(img_A).float()
        img_B = torch.from_numpy(img_B).float()
        label = torch.from_numpy(label).float()

        output = { # Returning the sample requested
            "img_A" : img_A,
            "img_B" : img_B,
            "label" : label
        }

        return  output
    

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def reshape_one_image (self, image :np.ndarray):
        """
        Arguments:
            image: shape = (H, W, C) or (H, W)
        """
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = ski_transform.resize(image, (new_h, new_w))

        return img

    def __call__(self, sample):
        label = sample["label"]
        img_A, img_B = sample["img_A"] , sample["img_B"]

        img_A = self.reshape_one_image(img_A)
        img_B = self.reshape_one_image(img_B)
        label = self.reshape_one_image(label)

        output = { # Returning the sample requested
            "img_A" : img_A,
            "img_B" : img_B,
            "label" : label
        }

        return  output
    

class Rotate(object):
    """
    Roatates image_A, image_B and label all by the same degree which is randomly selected...
    """
    def __call__(self, sample, *args: Any, **kwds: Any) -> Any:

        rand_rot_deg = np.random.randint(0, 360)

        img_A = sample["img_A"]
        img_B = sample["img_B"]
        label = sample["label"]

        img_A = ski_transform.rotate(img_A, rand_rot_deg)
        img_B = ski_transform.rotate(img_B, rand_rot_deg)
        label = ski_transform.rotate(label, rand_rot_deg)

        output = {
            "img_A" : img_A,
            "img_B" : img_B,
            "label" : label,
        }

        return output


class RandomNoise(object):
    def __call__(self, sample, *args: Any, **kwds: Any) -> Any:

        img_A = sample["img_A"]
        img_B = sample["img_B"]
        label = sample["label"]

        img_A = util.random_noise(img_A)
        img_B = util.random_noise(img_B)

        output = {
            "img_A" : img_A,
            "img_B" : img_B,
            "label" : label,
        }

        return output


class RandomBrightness(object):
    def __call__(self, sample, *args: Any, **kwds: Any) -> Any:

        img_A = sample["img_A"]
        img_B = sample["img_B"]
        label = sample["label"]

        num_1 = np.random.random()*1 + 1.0
        num_2 = np.random.random()*1 + 1.0

        img_A = exposure.adjust_gamma(img_A, num_1)
        img_B = exposure.adjust_gamma(img_B, num_2)

        output = {
            "img_A" : img_A,
            "img_B" : img_B,
            "label" : label,
        }

        return output       


def tensor_to_image(img_tensor :torch.Tensor) -> np.ndarray:
    
    # print("img_tensor type = ", type(img_tensor))
    # print("img_tensor shape = ", img_tensor.shape)
    
    img =  np.moveaxis(img_tensor.detach().clone().numpy() , 0, 2)
    if img.shape[2] == 1:
        img = img.reshape((img.shape[0], img.shape[1]))
    return img


def tensor_to_image_batch(img_tensor_batch :torch.Tensor) -> np.ndarray:
    img =  np.moveaxis(img_tensor_batch.detach().clone().numpy() , 1, -1)
    if img.shape[2] == 1:
        img = img.reshape((img.shape[0], img.shape[1]))
    return img


test_dataset = BuildingChangeDataset_Levir( # Final Dataset with all the pre-processing attached...
    "../../../Datasets/LEVIR-CD/images_split/test", 
    transforms = torchvis_transforms.Compose([
        Rescale(256),
        ToTensor(),
        SplitImageLabel()
    ])
)

train_dataset = BuildingChangeDataset_Levir( # Final Dataset with all the pre-processing attached...
    "../../../Datasets/LEVIR-CD/images_split/train",
    transforms= torchvis_transforms.Compose([
        RandomBrightness(),
        RandomNoise(),
        Rotate(),
        Rescale(256),
        ToTensor(),
        SplitImageLabel()
    ])
)

val_dataset = BuildingChangeDataset_Levir( # Final Dataset with all the pre-processing attached...
    "../../../Datasets/LEVIR-CD/images_split/val",
    transforms= torchvis_transforms.Compose([
        Rescale(256),
        ToTensor(),
        SplitImageLabel()
    ])
)

# train_size = int(0.8*(len(final_dataset)))
# test_size  = len(final_dataset) - train_size
# train_dataset, test_dataset = random_split(final_dataset, [train_size, test_size])



# final_dataloader = DataLoader(final_dataset, batch_size=8, shuffle=True, num_workers=0)


"""
l = np.random.randint(0, len(final_dataset)-3)

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 10))

cols = ["Image A", "Image B", "Label"]
# rows = ['Row {}'.format(row) for row in ['A', 'B', 'C']]
for ax, col in zip(axes[0], cols):
    ax.set_title(col)
# for ax, row in zip(axes[:,0], rows):
#     ax.set_ylabel(row, rotation=0, size='large')

fig.tight_layout()

for i in range( l , len(final_dataset)):
    sample = final_dataset[i]

    plt.subplot(3, 3, 3*(i-l) + 1)
    plt.imshow(tensor_to_image(sample["img_A"]))

    plt.subplot(3, 3, 3*(i-l) + 2)
    plt.imshow(tensor_to_image(sample["img_B"]))

    plt.subplot(3, 3, 3*(i-l) + 3)
    plt.imshow(tensor_to_image(sample["label"]), cmap="gray")    

    if i == l+2:
        break
        
"""