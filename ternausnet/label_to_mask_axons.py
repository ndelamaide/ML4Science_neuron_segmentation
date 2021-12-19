import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import closing, star


""" Converts labeled images to masks (two classes: neurons and axons)
    This assumes that we already have the masks for the neurons """

def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--path_labels', default="ternausnet/data/labels_unordered_axons/", type=str)
    arg('--path_binary_masks', default="ternausnet/data/masks/", type=str)

    args = parser.parse_args()

    root_path = os.path.join(os.pardir, args.path_labels)
    masks_path = os.path.join(os.pardir, args.path_binary_masks)

    save_path = os.path.join(os.pardir, "ternausnet/data/masks_with_axons/")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    factor = 127

    i = 1

    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".jpg"):

                image = cv2.imread(os.path.join(root, file), cv2.IMREAD_UNCHANGED)
                mask_binary = cv2.imread(os.path.join(masks_path, file.replace("label", "mask")), cv2.IMREAD_UNCHANGED)

                mask = np.zeros(mask_binary.shape)

                neurons = (mask_binary == 255)

                # Create masks for axons indexing on the RGB channels
                axons_r = (image[:, :, 0] > 65 ) & (image[:, :, 0] <= 75 )
                axons_g = (image[:, :, 1] > 68 ) & (image[:, :, 1] <= 85)
                axons_b = (image[:, :, 2] >= 245 )
                axons = axons_r & axons_g & axons_b
                
                # First set axons so they don't overlap with neurons
                mask[axons] = 2
                mask[neurons] = 1 

                mask = (mask * factor).astype(np.uint8)

                img_mask = closing(mask, star(5))

                cv2.imwrite(os.path.join(save_path, file.replace("label", "mask")), img_mask)

if __name__ == '__main__':
    main()