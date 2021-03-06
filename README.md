# Image-super-resolution
This is the image super resolution using SRResNet.

## Enviroment setting and dependencies 
Use pip install or conda install, and check the version :
```
#Name                        Version
python                       3.7.11
torch                        1.8.0
torchvision                  0.9.0
opencv-python                4.5.4.60
```

## Dataset 
There are 291 high resolution images for training and 14 low resolution images for testing, unscale factor=3.

## Code 
### 0. Download Project
```
git clone https://github.com/YLingT/Image-super-resolution  
cd Image-super-resolution
```
### 1.  Data preparing
The project structure are as follows:
```
Image-super-resolution
|── training_hr_images
|── testing_lr_images
|── testing_hr_images
|── img
|── create_data_lists.py
|── dataset.py
|── models.py
|── utils.py
|── train_srresnet.py
|── eval_x3.py
|── eval_x4.py
```
### 2.  Training
```
python train_srresnet.py
```

### 3.  Testing
Download trained weight: [checkpoint_srresnet.pth.tar](https://drive.google.com/file/d/1KFT_lzVbmm-b5fn799pUBXRaSP--8TBC/view?usp=sharing), put it under main folder.
```
python eval_x3.py
```

### 4.  Result analysis
Get PSNR = 27.6794
|   Original lr image  |  Upscale x3  |
|   :---:  |   :---:   |
|![04.png](https://github.com/YLingT/Image-super-resolution/blob/main/img/04.png)|![04_pred.png](https://github.com/YLingT/Image-super-resolution/blob/main/img/04_pred.png)|
|![11.png](https://github.com/YLingT/Image-super-resolution/blob/main/img/11.png)|![11_pred.png](https://github.com/YLingT/Image-super-resolution/blob/main/img/11_pred.png)|
|![12.png](https://github.com/YLingT/Image-super-resolution/blob/main/img/12.png)|![12_pred.png](https://github.com/YLingT/Image-super-resolution/blob/main/img/12_pred.png)|

### 5.  Reference
[a-PyTorch-Tutorial-to-Super-Resolution](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution)





