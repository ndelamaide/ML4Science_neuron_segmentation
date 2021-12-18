import numpy as np
from torch.utils.data import DataLoader
from dataset import CellsDataset
import torch
from tqdm import tqdm
import cv2
from PIL import Image
from utils import overlay



def predict(model, from_file_names, to_path, img_transform, num_classes=1):

    loader = DataLoader(
        dataset=CellsDataset(from_file_names, transform=img_transform, mode='predict'),
        shuffle=False
    )

    if num_classes > 1:

        factor = 127
    
    else:
        factor = 255

    num = 0
    with torch.no_grad():
        for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):

            outputs = model(inputs)

            for i, image_name in enumerate(paths):

                if num_classes > 1:

                    t_mask = (outputs[i].data.cpu().numpy().argmax(axis=0) * factor).astype(np.uint8)

                else:

                    t_mask = torch.sigmoid(outputs[i, 0]).data.cpu().numpy()

                    t_mask[t_mask >= 0.3] = 1
                    t_mask[t_mask < 0.3] = 0
                    t_mask = (t_mask * factor).astype(np.uint8)

                image = np.asarray(Image.open(image_name))
                image = np.moveaxis(np.array([image, image, image]), 0, -1).astype(np.uint8)

                img_overlay = overlay(image, t_mask, num_classes)
                
                name = image_name.split("/")[-1][:-4]

                cv2.imwrite(to_path + name + '_overlay.png', img_overlay)
                cv2.imwrite(to_path + name + "_mask.png", t_mask)
                num += 1