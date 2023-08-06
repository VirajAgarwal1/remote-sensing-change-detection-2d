import matplotlib.pyplot as plt
from torchvision.io.image import read_image
from networks import GenerateNet
from torchcam.methods import SmoothGradCAMpp
from dataloader import val_dataset, tensor_to_image
from torch.utils.data import DataLoader
import torchvision, torch


class Config():
    def __init__(self):
        self.MODEL_NAME = 'FCCDN'
        self.MODEL_OUTPUT_STRIDE = 16
        self.BAND_NUM = 3
        self.USE_SE = True

cfg = Config()
model = GenerateNet(cfg)


# Get your input
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
checkpoint = torch.load("./training/FCCDN_plain/best_model/best_model.pt", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])



with SmoothGradCAMpp(model) as cam_extractor:
  # Preprocess your data and feed it to the model
  out = model(input)
  # Retrieve the CAM by passing the class index and the model output
  activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)


cam_extractor = SmoothGradCAMpp(model)