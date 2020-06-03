from style_reconstruction import *
from utils import *
from pathlib import Path
from torchvision import models

Path('./export').mkdir(exist_ok=True)

style_weights = {
                    'conv1_1': 1.0,
                    'conv2_1': 0.8,
                    'conv3_1': 0.5,
                    'conv4_1': 0.3,
                    'conv5_1': 0.1
                }

img_path = 'https://i.pinimg.com/474x/f1/aa/82/f1aa820d303253b417a797f08fb2be67.jpg'

vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

recon_style(img_path, device, vgg, 'out', style_weights)