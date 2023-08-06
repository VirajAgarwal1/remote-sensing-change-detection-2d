import torch
import numpy as np
from torch import optim
import torchvision
from torch.utils.data import DataLoader
from loss import FCCDN_loss_BCD
import shutil
import os
from networks import GenerateNet
from dataloader import train_dataset, test_dataset, val_dataset, tensor_to_image
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXP_NAME = "/FCCDN_plain"
BATCH_SIZE = 20
NUM_EPOCHS = 100
THRESHOLD_HYP_PARAM = 0.7

# define model
class Config():
    def __init__(self):
        self.MODEL_NAME = 'FCCDN'
        self.MODEL_OUTPUT_STRIDE = 16
        self.BAND_NUM = 3
        self.USE_SE = True
cfg = Config()
model = GenerateNet(cfg)
# define loss function and optimizer
criterion = FCCDN_loss_BCD
optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=0.001)


# python3 -m tensorboard.main --logdir=runs
writer = SummaryWriter( "runs/FCCDN" + EXP_NAME )


def mcc_calc (conf_mat) :

    tp, tn, fp, fn = conf_mat

    mcc = tn*tp - fn*fp
    den = torch.sqrt( (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) )
    if not den:
        den = 1
    mcc = mcc / den

    return mcc


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
        # Mathews Corelation Coefficient
        train_mcc = 0.0
        train_conf_mat = [0.0, 0.0, 0.0, 0.0] # TP, TN, FP, FN
        valid_mcc = 0.0
        valid_conf_mat = [0.0, 0.0, 0.0, 0.0] # TP, TN, FP, FN

        torch.cuda.empty_cache()

        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, sample in enumerate(loaders['train']):

            # Move input tensors to the device
            img_A = sample["img_A"].clone().to(device)
            img_B = sample["img_B"].clone().to(device)
            label1 = sample["label"].clone().to(device)

            # find the loss and update the model parameters accordingly
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model([img_A.clone(), img_B.clone()])
            # calculate the batch loss
            label2 = torchvision.transforms.Resize(sample["label"].shape[2] // 2, antialias=False)(label1).clone().to(device)
            label = [label1.clone().float(), label2.clone().float()]
            loss = criterion(output, label)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record the average training loss, using something like
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            # Confusion Matrix calculation
            out_thres = torch.where(output[0] >= THRESHOLD_HYP_PARAM, 1, 0)
            equal_mat = out_thres == label1  # Compare with label1, which is on the same device
            not_equal_mat = out_thres != label1
            train_conf_mat[0] += torch.sum(torch.where(out_thres == 1, equal_mat, 0))  # TP
            train_conf_mat[1] += torch.sum(torch.where(out_thres == 0, equal_mat, 0))  # TN
            train_conf_mat[2] += torch.sum(torch.where(out_thres == 1, not_equal_mat, 0))  # FP
            train_conf_mat[3] += torch.sum(torch.where(out_thres == 0, not_equal_mat, 0))  # FN
            batch_mcc = mcc_calc(train_conf_mat)

            print("Epoch: {:4d}/{:4d}\tBatch: {:4d}/{:4d}\tTraining Loss: {:.3f}\tTraining MCC: {:.3f}\t\t\t\t".format(
                epoch, 
                n_epochs, 
                batch_idx+1, 
                len(loaders["train"]), 
                train_loss,
                batch_mcc
            ), end="\r")

            if batch_idx % 20 == 0:
                writer.add_image( "Prediction",    output[0][0],       epoch*(len(loaders['train'])) + batch_idx + 1)
                writer.add_image( "Target",        sample["label"][0], epoch*(len(loaders['train'])) + batch_idx + 1)
                writer.add_scalar("Training Loss", train_loss,         epoch*(len(loaders['train'])) + batch_idx + 1)
                writer.add_scalar("Training MCC",  batch_mcc,          epoch*(len(loaders['train'])) + batch_idx + 1)
                
        print("\n", end="")
        
        torch.cuda.empty_cache()

        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, sample in enumerate(loaders['val']):

            # Move input tensors to the device
            img_A = sample["img_A"].clone().to(device)
            img_B = sample["img_B"].clone().to(device)
            label1 = sample["label"].clone().to(device)

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model([img_A.clone(), img_B.clone()])
            # calculate the batch loss
            label2 = torchvision.transforms.Resize(sample["label"].shape[2] // 2, antialias=False)(label1).clone().to(device)
            label = [label1.clone().float(), label2.clone().float()]
            loss = criterion(output, label)

            # update average validation loss
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            # Confusion Matrix calculation
            out_thres = torch.where(output[0] >= THRESHOLD_HYP_PARAM, 1, 0)
            equal_mat = out_thres == label1  # Compare with label1, which is on the same device
            not_equal_mat = out_thres != label1
            valid_conf_mat[0] += torch.sum(torch.where(out_thres == 1, equal_mat, 0))  # TP
            valid_conf_mat[1] += torch.sum(torch.where(out_thres == 0, equal_mat, 0))  # TN
            valid_conf_mat[2] += torch.sum(torch.where(out_thres == 1, not_equal_mat, 0))  # FP
            valid_conf_mat[3] += torch.sum(torch.where(out_thres == 0, not_equal_mat, 0))  # FN
            batch_mcc = mcc_calc(valid_conf_mat)

            print("Epoch: {:4d}/{:4d}\tBatch: {:4d}/{:4d}\tValidation Loss: {:.3f}\tValidation MCC: {:.3f}\t\t\t\t".format(
                epoch,
                n_epochs,
                batch_idx + 1,
                len(loaders["val"]),
                valid_loss,
                batch_mcc
            ), end="\r")

            if batch_idx % 20 == 0:
                writer.add_scalar("Valid Loss", valid_loss, epoch * (len(loaders['val'])) + batch_idx + 1)
                writer.add_scalar("Valid MCC", batch_mcc, epoch * (len(loaders['val'])) + batch_idx + 1)
                writer.add_image("Prediction", output[0][0], epoch * (len(loaders['val'])) + batch_idx + 1)
                writer.add_image("Target", sample["label"][0], epoch * (len(loaders['val'])) + batch_idx + 1)

        print("\n", end="")



        # calculate average losses
        train_mcc = mcc_calc(conf_mat=train_conf_mat)
        valid_mcc = mcc_calc(conf_mat=valid_conf_mat)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTrain MCC: {:.6f} \tValid MCC: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss,
            train_mcc,
            valid_mcc
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


training_model_save_path = "./training" + EXP_NAME
if not os.path.exists(training_model_save_path):
    os.mkdir(training_model_save_path)
if not os.path.exists(training_model_save_path + "/checkpoint/"):
    os.mkdir(training_model_save_path + "/checkpoint/")
if not os.path.exists(training_model_save_path + "/best_model/"):
    os.mkdir(training_model_save_path + "/best_model/")

# trained_model = train(
#     0, 
#     NUM_EPOCHS, 
#     np.Inf, 
#     loaders, 
#     model, 
#     optimizer,  
#     criterion, 
#     False, 
#     training_model_save_path + "/checkpoint/current_checkpoint.pt", 
#     training_model_save_path + "/best_model/best_model.pt"
# )

model, optimizer, start_epoch, valid_loss_min = load_ckp(
    training_model_save_path + "/checkpoint/current_checkpoint.pt", 
    model, 
    optimizer
)

model = model.to(device)
# optimizer = optimizer.to(device)
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

trained_model = train(
    start_epoch, 
    NUM_EPOCHS, 
    valid_loss_min, 
    loaders, 
    model, 
    optimizer, 
    criterion, 
    False, 
    training_model_save_path + "/checkpoint/current_checkpoint.pt", 
    training_model_save_path + "/best_model/best_model.pt"
)

writer.close()
