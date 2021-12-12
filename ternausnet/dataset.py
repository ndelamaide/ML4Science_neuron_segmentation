from numpy.lib.type_check import imag
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import os
from albumentations.pytorch.functional import img_to_tensor

def get_file_names(folder_name):

    data_base_path = os.path.join(os.pardir, 'ternausnet/'+folder_name)
    names = []

    for root, dirs, files in os.walk(data_base_path):
        for file in files:
            if file.endswith(".tif"):

                names.append(os.path.join(root, file))
    
    return names

class CellsDataset(Dataset):
    def __init__(self, file_names, to_augment=False, transform=None, mode='train'):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name)

        data = {"image": image, "mask": mask}
        if self.transform:
            augmented = self.transform(**data)
            image, mask = augmented["image"], augmented["mask"]
        else:
            image, mask = data["image"], data["mask"]

        if self.mode == 'train':
            return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
        else:
            return img_to_tensor(image), str(img_file_name)

def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    
    mask_folder = "masks"

    mask = cv2.imread(str(path).replace('images', mask_folder).replace('image', 'mask').replace(".tif", ".jpg"), cv2.IMREAD_UNCHANGED )
    mask[mask != 255] = 0
    
    factor = 255

    return (mask / factor).astype(np.uint8)