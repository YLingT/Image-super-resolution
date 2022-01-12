import glob
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
import PIL
from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
image_list = glob.glob("testing_lr_images/*.png")

# Model checkpoints
srresnet_checkpoint = "checkpoint_srresnet.pth.tar"

# Load SRResNet model
srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()
model = srresnet

imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)     

with torch.no_grad():
    for image_name in image_list:        # Move to default device
        print(image_name)
        lr_imgs = Image.open(image_name)
        lr_imgs = lr_imgs.convert('RGB')
        lr_imgs = convert_image(lr_imgs, source='pil', target='imagenet-norm')
        
        sr_imgs = convert_image(model(lr_imgs.unsqueeze(0).to(device)).squeeze(0).cpu().detach(), '[-1, 1]', 'pil')
        sr_imgs = sr_imgs.resize((3 * sr_imgs.size[0] // 4, 3 * sr_imgs.size[1] // 4))
        na = os.path.basename(image_name).replace('.png','')
        sr_imgs.save(os.path.join('testing_hr_images', na + '_pred.png'))
