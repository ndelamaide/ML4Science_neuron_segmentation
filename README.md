# ML4Science for the QLAB at EPFL

The goal of the project was to perform image segmentation on images of neuronal cells growing on a diamond.

We implemented two variants of the TernausNet model : one is a U-Net with a VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation, and the other uses VGG16 instead of VGG11 ([arxiv paper](https://arxiv.org/abs/1801.05746)).
The GitHub for the TernausNet models can be found [here](https://github.com/ternaus/TernausNet).

The lab gave us several images that we had to label by hand. As a first step, we only labeled the neurons and trained our models on those images. As a second step, we also labeled the axons and trained our models to segment both the neurons and axons.

## Data and models

**TODO**
The original images and their labels can be found on this [link]().
The weights of our models, UNet11 and UNet16 for neurons only and for neurons and axons can be found on this [link]().

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

In the end, to use our models for neuron segmentation only just run `data_job_binary.sh`, and for neuron and axon segmentation run `data_job_axons.sh` as well afterwards.

# Training and validation

To train our model, simply run `train.py`. Here is a list of useful arguments:

- `--model` : the model to use, either UNet11 or UNet16.
- `--num_classes` : the number of classes. For neurons only, use a value of 1. For neurons and axons, use a value of 3.
- `--batch_size` : the batch size, which is 4 by default. For num_classes > 1, it is set to 1.
- `--epochs` : the number of epochs to train the model for.
- `--checkpoint_path` : the path to load the model from. Only needed when using resuming training from a checkpoint. The path need to start by *ternausnet/*.
- `--save_checkpoint_name` : the name with which to save the model at the end of each training epoch, for example *unet16.pt*. It needs to end with either *.pt* or *.pth*


To load our data for training and validation, we use dataloaders that respectivaly perform the following operations:
- Randomly crop the images to 512 x 768, perform a vertical flip with probability 0.5, perform a horizontal flip with probability 0.5, then normalize the image.
- Or, perform a center crop to 512 x 768 and then normalize the image.

After each training epoch, we perform validation on the validation data and record the loss and jaccard on the validation set. Then we save the model checkpoint, the losses so far as well as the jaccard values.

# Evaluation

To evaluate our model, run `test.py`. Just like for training, here is a list of useful arguments :

- `--model` : the model to use, either UNet11 or UNet16.
- `--num_classes` : the number of classes. For neurons only, use a value of 1. For neurons and axons, use a value of 3.
- `--checkpoint_path` : the path to load the model from. The path need to start by *ternausnet/*.
- `--file_names_test` : the relative path to the images to evaluate on, by default it's *test_data/images*.

