# ML4Science for the QLAB at EPFL

The goal of the project was to perform image segmentation on images of neuronal cells growing on a diamond.

We implemented two variants of the TernausNet model : one is a U-Net with a VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation, and the other uses VGG16 instead of VGG11 [arxiv paper](https://arxiv.org/abs/1801.05746).

The lab gave us several images that we had to label by hand. As a first step, we only labeled the neurons and trained our models on those images. As a second step, we also labeled the axons and trained our models to segment both the neurons and axons.

## Pre-processing

The images from the lab came in a folder structured like this : 

- originals
  - 200_2
  - 200_4
  - 400_2
  - 400_4
  - empty

We labeled the images and saved them in a folder following the same structure called *labels*. We also removed duplicate images to obtain a total of 41 images.
We used the script `data_job_binary.sh` which converts the original images and the labeled images to a train - test - validation split (0.7% - 0.1 % - 0.2 %) of the original images and their masks (neurons only).
To do so it creates two new folders *data/labels_unordered* and *data/originals_unordered* which contain the labeled and original images respectively numbered from 1 to 41. Then it creates the masks from the labels and finaly creates the split by creating three folders *train_data*, *val_data*, *test_data* structured like this:

- folder_data
  - images
  
     image_x.tif
     
     ....
  - masks
  
     mask_x.jpg
     
     ....
 
Moreover each of the 41 images and their masks of size 1536 x 2048 where divided in 4 images of size 768 x 1024 by cropping the original images and masks.

For the labeled axons, we used the script `data_job_axons.sh` which creates the masks for the axons and adds them to the pre-existing masks of the neurons. We thus have a mask for two classes : the neurons and axons (as well as background). Then it created the same train - test - validation split as before but with one extra folder in each split : *masks_with_axons*.

# Training and validation


# Evaluation
