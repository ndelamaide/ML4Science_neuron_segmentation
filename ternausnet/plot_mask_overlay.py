import argparse
import matplotlib.pyplot as plt
from utils import overlay
import cv2
import os
import numpy as np

""" Plots the ground truth and the predicted overlays of the models for a given image
    Used for the plots in the report. """

def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--path_image', default="ternausnet/test_data/images/image_1.tif", type=str)
    arg('--path_overlay_11', type=str) # overlay of unet11
    arg('--path_overlay_16', type=str) #overlay of unet16
    arg('--num_classes', default=1, type=int)

    args = parser.parse_args()

    path_image = os.path.join(os.pardir, args.path_image)

    if args.num_classes > 1:
        path_ground_truth = os.path.join(os.pardir, args.path_image.replace("images", "masks_with_axons").replace("image", "mask").replace("tif", "jpg"))
    else:
        path_ground_truth = os.path.join(os.pardir, args.path_image.replace("images", "masks").replace("image", "mask").replace("tif", "jpg"))


    image = cv2.cvtColor(cv2.imread(path_image), cv2.COLOR_BGR2RGB)
    ground_truth = cv2.imread(path_ground_truth, cv2.IMREAD_UNCHANGED)

    ground_truth_overlay = overlay(image, ground_truth, num_classes=args.num_classes)
    overlay_11 = cv2.cvtColor(cv2.imread(os.path.join(os.pardir, args.path_overlay_11), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    overlay_16 =cv2.cvtColor(cv2.imread(os.path.join(os.pardir, args.path_overlay_16), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
     
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    axes[0].imshow(ground_truth_overlay)
    axes[1].imshow(overlay_11)
    axes[2].imshow(overlay_16)

    titles = ['Ground Truth', 'UNet11', 'Unet16']
    for i in range(len(axes)):
        axes[i].set_axis_off()
        axes[i].title.set_text(titles[i])

    fig_folder = os.path.join(os.pardir, "ternausnet/figs/")
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    image_name = path_image.split("/")[-1][:-4]
    plt.tight_layout()
    plt.savefig(fig_folder+image_name+"_fig.jpg")
    plt.show()

if __name__ == '__main__':
    main()