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
torch.manual_seed(17)

# create csv file with all images as filename
import os
import csv
#data = pd.DataFrame()
images = []
for folder in os.listdir("/home/james/neuronDetection/data/"):
    if not folder.startswith('.'):
        for filename in os.listdir("/home/james/neuronDetection/data/"+folder):
            if filename.endswith(".tif"):
                images.append("/home/james/neuronDetection/data/"+folder+"/"+filename)
            else:
                continue
        #data[folder] = list

print(len(images))
print(images[68])
print(images[69])

augmented_data = pd.DataFrame()
transform = AffineTransform(translation=(25,25))
sigma=0.155

for i in tqdm(range(len(images))):
    image_transforms = []
    image = imread(images[i])
    image_transforms.append(rotate(image, angle=45, mode = 'wrap'))
    image_transforms.append(warp(image, transform, mode='wrap'))
    image_transforms.append(np.fliplr(image))
    image_transforms.append(np.flipud(image))
    image_transforms.append(random_noise(image, var=sigma**2))
    image_transforms.append(gaussian(image,sigma=1,multichannel=True))

    #for transform in image_transforms :
    #    plt.imshow(transform)
    #    plt.show()
    augmented_data[images[i]] = image_transforms
    # la dataframe devient trop grande donc on peut pas la process -> save as tif image ?

print(augmented_data)
