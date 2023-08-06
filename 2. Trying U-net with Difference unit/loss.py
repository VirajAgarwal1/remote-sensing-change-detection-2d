import os
from typing import Any
from skimage import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# # We need information on how many images have changes in them and how many dont...
# dir_path = "../../Dataset Findings/LEVIR-CD/images_split/label"
# imgs_dir = os.listdir(dir_path)
# num_imgs_train = len(imgs_dir)

# num_change_pixesl_imgs = 0
# num_no_change_pixel_imgs = 0

# for img_name in imgs_dir:
#     img = io.imread(os.path.join(dir_path, img_name), as_gray=True)
#     img[img >= 0.7] = 1
#     img[img < 0.7] = 0
#     sum_img = img.sum()
#     num_change_pixesl_imgs += sum_img
#     num_no_change_pixel_imgs += (256*256) - sum_img    
    

# f = open("training_imgs_info.txt", "a")
# f.write("Number of pixels in images with change = {}\nNumber of pixels in images with no change = {}".format(
#     num_change_pixesl_imgs, num_no_change_pixel_imgs
# ))
# f.close()


class WeightedBCE (nn.Module):
    def __init__(self, pos_weight) -> None:
        super().__init__()
        self.pos_weight = pos_weight

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

class CustomBinFocalLoss (nn.Module):
    """
    Computes Focal Loss for Binary classification. This takes a batch of data and computes loss for it.
    
    Arguments:
        alpha : float, default = 0.8, The more the alpha the more the contribution of 1s in the loss. 
                So class 1 will have alpha times less impact in the loss. Should be equal to (1 - (0's pixels / 1's pixels))
        
        gamma : float, default = 2, Influences how much does the easy classifications contribute... 
                Predictions which are already close to their expected value...
    """
    def __init__(self, alpha = 0.8, gamma = 2) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cross_entropy = nn.BCELoss(reduction="none")

    def __call__(self, pred :torch.Tensor, target :torch.Tensor) -> torch.Tensor:

        # ce = torch.where( target==1 , torch.log(pred) , torch.log(1-pred) )
        cross_entropy = self.cross_entropy( pred, target)

        pt = torch.where( target==1 , self.alpha*(1-pred) , (1-self.alpha)*pred )
        pt = pt**self.gamma
        
        loss = pt * cross_entropy
        # loss = torch.mean(loss)   
        loss = torch.sum(loss)   
        return loss
    
class DiceLoss (nn.Module):
    
    def __init__(self) -> None:
        super().__init__()


    def forward (self, logits, true, eps=1e-7):
        """Computes the Sørensen–Dice loss.

        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.

        Args:
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            true: a tensor of shape [B, 1, H, W].
            eps: added to the denominator for numerical stability.

        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)
    
