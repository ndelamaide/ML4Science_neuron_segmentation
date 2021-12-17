import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import closing, star

""" Converts labeled images to masks (binary) """

def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--root', default="ternausnet/data/labels_unordered", type=str)

    args = parser.parse_args()

    root_path = os.path.join(os.pardir, args.root)

    save_path = os.path.join(os.pardir, "ternausnet/data/masks/")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".jpg"):

                image = cv2.imread(os.path.join(root, file), cv2.IMREAD_UNCHANGED)

                # Together cells
                mask_1 = (image[:, :, 0] >= 250)
                mask_2 = (image[:, :, 1] >= 48) & (image[:, :, 1] <= 55)
                mask_3 = (image[:, :, 2] >= 250)
                mask = mask_1 & mask_2 & mask_3

                img_mask = np.zeros((image.shape[0], image.shape[1]))

                img_mask[mask] = 255

                # Alone cells
                mask_1 = (image[:, :, 0] >= 250)
                mask_2 = (image[:, :, 1] >= 225) & (image[:, :, 1] <= 235)
                mask_3 = (image[:, :, 2] >= 25) & (image[:, :, 2] <= 35)
                mask = mask_1 & mask_2 & mask_3

                img_mask[mask] = 255

                img_mask = closing(img_mask, star(5))

                cv2.imwrite(os.path.join(save_path, file.replace("label", "mask")), img_mask)

if __name__ == '__main__':
    main()