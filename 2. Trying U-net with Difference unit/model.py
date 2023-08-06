from torch import subtract, abs, concat
from torch import nn

class UNetwork (nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d( 3,  16, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16,  32, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32,  64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=False)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2, padding=0, bias=False)
        self.deconv3 = nn.ConvTranspose2d( 64, 16, kernel_size=2, stride=2, padding=0, bias=False)
        self.deconv4 = nn.ConvTranspose2d( 32,  1, kernel_size=2, stride=2, padding=0, bias=False)

        self.bNorm1 = nn.BatchNorm2d( 16)
        self.bNorm2 = nn.BatchNorm2d( 32)
        self.bNorm3 = nn.BatchNorm2d( 64)
        self.bNorm4 = nn.BatchNorm2d(128)
        self.bNorm5 = nn.BatchNorm2d( 64)
        self.bNorm6 = nn.BatchNorm2d( 32)
        self.bNorm7 = nn.BatchNorm2d( 16)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.act4 = nn.ReLU()
        self.act5 = nn.ReLU()
        self.act6 = nn.ReLU()
        self.act7 = nn.ReLU()

        self.sig = nn.Sigmoid()
        

        self.sub = subtract
        self.abs = abs
        self.concat = concat

        self.thres_hyp_param = 0.7 # Thresholding of the image will be part of the prediction but not training

    def forward_one_img (self, img):
        
        out = self.conv1(img)
        out = self.bNorm1(out)
        out = self.act1(out)
        x1 = out.clone() # Making clone so as to make a skip connection

        out = self.conv2(out)
        out = self.bNorm2(out)
        out = self.act2(out)
        x2 = out.clone() # Making clone so as to make a skip connection

        out = self.conv3(out)
        out = self.bNorm3(out)
        out = self.act3(out)
        x3 = out.clone() # Making clone so as to make a skip connection

        out = self.conv4(out)
        out = self.bNorm4(out)
        out = self.act4(out)


        out = self.deconv1(out)
        out = self.bNorm5(out)
        out = self.act5(out)
        # Connecting the skip connection from previous layers by concatenating them along the channels dimension
        out = self.concat((x3, out), dim=1) 

        out = self.deconv2(out) # In channels increased due to skip connection
        out = self.bNorm6(out)
        out = self.act6(out)
        # Connecting the skip connection from previous layers by concatenating them along the channels dimension
        out = self.concat((x2, out), dim=1) 

        out = self.deconv3(out) # In channels increased due to skip connection
        out = self.bNorm7(out)
        out = self.act7(out)
        # Connecting the skip connection from previous layers by concatenating them along the channels dimension
        out = self.concat((x1, out), dim=1) 

        out = self.deconv4(out) # In channels increased due to skip connection
        out = self.sig(out)

        return out

    def forward (self, img1, img2, threshold=True):
        """
            Input Images should be of size (256,256) and should be normalized to values between 0 and 1 only...
        """
        detection_img1 = self.forward_one_img(img1)
        detection_img2 = self.forward_one_img(img2)
        out = self.abs( self.sub( detection_img1 , detection_img2 ) )

        if threshold:
            # Thresholding the output 
            out[out >= self.thres_hyp_param] = 1
            out[out < self.thres_hyp_param] = 0
        
        return out
    
        

    
    






