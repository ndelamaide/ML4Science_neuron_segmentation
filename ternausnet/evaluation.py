import numpy as np
from torch.utils.data import DataLoader
from dataset import CellsDataset
import torch
from tqdm import tqdm
import cv2
from PIL import Image
from utils import overlay, get_jaccard
import os


def predict(model, from_file_names, to_path, img_transform, num_classes=1):

    loader = DataLoader(
        dataset=CellsDataset(from_file_names, transform=img_transform, mode='predict'),
        shuffle=False
    )

    overlays_folder = os.path.join(to_path, "overlays/")
    if not os.path.exists(overlays_folder):
        os.makedirs(overlays_folder)

    masks_folder = os.path.join(to_path, "masks/")
    if not os.path.exists(masks_folder):
        os.makedirs(masks_folder)

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

                cv2.imwrite(overlays_folder + name + '_overlay.jpg', cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR))
                cv2.imwrite(masks_folder + name + "_mask.jpg", t_mask)
                num += 1



def eval(from_file_names, to_path, num_classes):

    loader = DataLoader(
        dataset=CellsDataset(from_file_names, mode='predict'),
        shuffle=False
    )

    jaccard = []

    for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Eval')):
        for i, image_name in enumerate(paths):

            if num_classes > 1:
                masks_path = image_name.replace("images", "masks_with_axons").replace("image", "mask").replace(".tif", ".jpg")
            else:
                masks_path = image_name.replace("images", "masks").replace("image", "mask").replace(".tif", ".jpg")

            mask_name = masks_path.split("/")[-1][:-4]
            predicted_path = os.path.join(to_path, "masks/" + mask_name.replace("mask", "image") + "_mask.jpg")

            mask = cv2.imread(masks_path, cv2.IMREAD_UNCHANGED)
            predicted_mask = cv2.imread(predicted_path, cv2.IMREAD_UNCHANGED)

            if num_classes > 1:

                factor = 127
                target = (mask / factor).astype(np.uint8)
                output = (predicted_mask / factor).astype(np.uint8)

                target_neurons = np.zeros(target.shape)
                target_neurons[target == 1] = 1

                target_axons = np.zeros(target.shape)
                target_axons[target == 2] = 1

                jaccard_neurons = get_jaccard(target_neurons, (output == 1).astype(float))
                jaccard_axons = get_jaccard(target_axons, (output == 2).astype(float))

                jaccard.append([jaccard_neurons, jaccard_axons])
                
            else:

                factor = 255
                target = (mask / factor).astype(np.uint8)
                output = (predicted_mask / factor).astype(np.uint8)

                jaccard_ = get_jaccard(target, (output > 0).astype(float))

                jaccard.append(jaccard_)

    metrics_array = np.array(jaccard)
    mean = metrics_array.mean(axis=0)
    std = metrics_array.std(axis=0)
    max_ = metrics_array.max(axis=0)

    
    if num_classes > 1:
        print("Jaccard neurons | mean : {mean}, std : {std}, max: {max}".format(mean=mean[0], std=std[0], max=max_[0]))
        print("Jaccard axons | mean : {mean}, std : {std}, max: {max}".format(mean=mean[1], std=std[1], max=max_[1]))
    else:
        print("Jaccard neurons | mean : {mean}, std : {std}, max: {max}".format(mean=mean, std=std, max=max_))

    return jaccard