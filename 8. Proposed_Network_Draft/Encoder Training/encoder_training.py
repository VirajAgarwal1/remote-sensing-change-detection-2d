from utils.encoder import Encoder
import torch
import numpy as np
from torch import optim
import torchvision
from torch.utils.data import DataLoader
import shutil
import os
from dataloader import train_dataset, test_dataset, val_dataset, tensor_to_image
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 20
NUM_EPOCHS = 100
THRESHOLD_HYP_PARAM = 0.7
LR = 0.001
WEIGHT_DECAY = 0.001


# define model
model = Encoder("swin")
# define loss function and optimizer
criterion = torch.nn.CosineEmbeddingLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


# python3 -m tensorboard.main --logdir=runs
writer = SummaryWriter( "runs/Swin_Encoder")
model.eval()
sample = train_dataset[23]
writer.add_graph(model, [sample["img_A"].reshape((1,3,256,256)) , sample["img_B"].reshape((1,3,256,256))] )

def mcc_calc (conf_mat) :

    tp, tn, fp, fn = conf_mat

    mcc = tn*tp - fn*fp
    den = torch.sqrt( (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) )
    if not den:
        den = 1
    mcc = mcc / den

    return mcc


def f1_score_calc (conf_mat):

    tp, tn, fp, fn = conf_mat

    prec = tp / (tp + fp)
    rec  = tp / (tp + fn)

    f1 = (2*prec*rec) / (prec + rec)

    return f1


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)
    

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath, map_location=device)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


def train(start_epochs, n_epochs, valid_loss_min_input, loaders, model, optimizer, criterion, use_cuda, checkpoint_path, best_model_path):
    """
    Keyword arguments:
    start_epochs -- the real part (default 0.0)
    n_epochs -- the imaginary part (default 0.0)
    valid_loss_min_input
    loaders
    model
    optimizer
    criterion
    use_cuda
    checkpoint_path
    best_model_path

    returns trained model
    """
    model = model.to(device)
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    for epoch in range(start_epochs, 1+n_epochs):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        torch.cuda.empty_cache()

        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, sample in enumerate(loaders['train']):

            # Move input tensors to the device
            img_A = sample["img_A"].to(device)
            img_B = sample["img_B"].to(device)
            label = torch.flatten(sample["label"].to(device))

            # find the loss and update the model parameters accordingly
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            enc1, enc2 = model(img_A, img_B)
            # calculate the batch loss
            loss = criterion(enc1, enc2, label)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record the average training loss, using something like
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))


            print("Epoch: {:4d}/{:4d}\tBatch: {:4d}/{:4d}\tTraining Loss: \t\t\t".format(
                epoch, 
                n_epochs, 
                batch_idx+1, 
                len(loaders["train"]), 
                train_loss
            ), end="\r")

            if batch_idx % 20 == 0:
                writer.add_scalar("Training Loss", train_loss, epoch*(len(loaders['train'])) + batch_idx + 1)
            
                
        print("\n", end="")
        
        torch.cuda.empty_cache()

        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, sample in enumerate(loaders['val']):

            # Move input tensors to the device
            img_A = sample["img_A"].to(device)
            img_B = sample["img_B"].to(device)
            label = torch.flatten(sample["label"].to(device))

            # forward pass: compute predicted outputs by passing inputs to the model
            enc1, enc2 = model(img_A, img_B)
            # calculate the batch loss
            loss = criterion(enc1, enc2, label)
            # update average validation loss
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))


            print("Epoch: {:4d}/{:4d}\tBatch: {:4d}/{:4d}\tValidation Loss: {:.3f}\t\t\t\t".format(
                epoch,
                n_epochs,
                batch_idx + 1,
                len(loaders["val"]),
                valid_loss
            ), end="\r")

            if batch_idx % 20 == 0:
                writer.add_scalar("Valid Loss", valid_loss, epoch * (len(loaders['val'])) + batch_idx + 1)
            

        print("\n", end="")



        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \t'.format(
            epoch,
            train_loss,
            valid_loss
        ))

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss
        
        

    # return trained model
    return model


loaders = {
    'train':    DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
    'test':     DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True),
    'val':      DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True),
}


# for i, batch_sample in enumerate(loaders["train"]):
#     # print(batch_sample["img_A"].type)
#     input = torch.stack( [batch_sample["img_A"], batch_sample["img_B"]] )
#     writer.add_graph(model, input)
#     break


training_model_save_path = "./training"
if not os.path.exists(training_model_save_path):
    os.mkdir(training_model_save_path)
if not os.path.exists(training_model_save_path + "/checkpoint/"):
    os.mkdir(training_model_save_path + "/checkpoint/")
if not os.path.exists(training_model_save_path + "/best_model/"):
    os.mkdir(training_model_save_path + "/best_model/")

trained_model = train(
    0, 
    NUM_EPOCHS, 
    np.Inf, 
    loaders, 
    model, 
    optimizer,  
    criterion, 
    False, 
    training_model_save_path + "/checkpoint/current_checkpoint.pt", 
    training_model_save_path + "/best_model/best_model.pt"
)

### Continuing model training part of code....

# model, optimizer, start_epoch, valid_loss_min = load_ckp(
#     training_model_save_path + "/checkpoint/current_checkpoint.pt", 
#     model, 
#     optimizer
# )

# model = model.to(device)
# # optimizer = optimizer.to(device)
# for state in optimizer.state.values():
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.to(device)

# trained_model = train(
#     start_epoch, 
#     NUM_EPOCHS, 
#     valid_loss_min, 
#     loaders, 
#     model, 
#     optimizer, 
#     criterion, 
#     False, 
#     training_model_save_path + "/checkpoint/current_checkpoint.pt", 
#     training_model_save_path + "/best_model/best_model.pt"
# )

writer.close()
