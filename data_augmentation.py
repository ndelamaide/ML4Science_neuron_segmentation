# importing all the required libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
import matplotlib.pyplot as plt

#from torchsummary import summary
import pandas as pd
from skimage.io import imread, imsave
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from scipy import ndimage
#torch.manual_seed(17)

# create csv file with all images as filename
import os
import csv
#data = pd.DataFrame()
mypath = "/home/james/LIVECell/images/sliced/unlabelled/all"
images = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

transform = AffineTransform(translation=(25,25))
sigma=0.155

for i in tqdm(range(len(images))):
    #image_transforms = []
    image = imread(os.path.join(mypath, images[i]))
    #image_transforms.append(rotate(image, angle=45, mode = 'wrap'))
    #image_transforms.append(warp(image, transform, mode='wrap'))
    #image_transforms.append(np.fliplr(image))
    #image_transforms.append(np.flipud(image))
    #image_transforms.append(random_noise(image, var=sigma**2))
    #image_transforms.append(gaussian(image,sigma=1,multichannel=True))
    name_save = [os.path.splitext(images[i])[0].lower() + "_" + str(j) + ".jpg" for j in range(6)]
    imsave(os.path.join(mypath, name_save[0]), rotate(image, angle=45, mode = 'wrap'))
    imsave(os.path.join(mypath, name_save[1]), warp(image, transform, mode='wrap'))
    imsave(os.path.join(mypath, name_save[2]), np.fliplr(image))
    imsave(os.path.join(mypath, name_save[3]), np.flipud(image))
    imsave(os.path.join(mypath, name_save[4]), random_noise(image, var=sigma**2))
    imsave(os.path.join(mypath, name_save[5]), gaussian(image,sigma=1,multichannel=True))

    #for transform in image_transforms :
    #    plt.imshow(transform)
    #    plt.show()
    #augmented_data[images[i]] = image_transforms
    # la dataframe devient trop grande donc on peut pas la process -> save as tif image ?
