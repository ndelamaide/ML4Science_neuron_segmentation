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

    arg('--path_labels', default="ternausnet/data/labels_unordered", type=str)
    arg('--path_binary_masks', default="ternauset/data/masks", type=str)

    args = parser.parse_args()

    root_path = os.path.join(os.pardir, args.path_labels)

    save_path = os.path.join(os.pardir, "ternauset/data/masks_with_axons")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    factor = 127

    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".jpg"):

                image = cv2.imread(os.path.join(root, file), cv2.IMREAD_UNCHANGED)
                mask = cv2.imread(os.path.join(root, file.replace("label", "mask")), cv2.IMREAD_UNCHANGED)
                mask[mask==255] = 1



                ## TODO : EXTRACT AXON LABELS FROM IMAGE
                ## then morpho, then add to original mask
                ## use neurons = 1 axons = 2 then multiply by factor


                mask_axons = np.zeros((image.shape[0], image.shape[1]))


                img_mask = closing(img_mask, star(5))

                cv2.imwrite(os.path.join(save_path, file.replace("label", "mask")), img_mask)

if __name__ == '__main__':
    main()