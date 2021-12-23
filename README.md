# ML4Science for the QLAB at EPFL

The goal of the project was to perform image segmentation on images of neuronal cells growing on a diamond.

You can look at our specific work about image analysis in the folder image_analysis and the LIVECell adaptaion in the livecell directory, but the main results of the working model is explained below. 

We implemented two variants of the TernausNet model : one is a U-Net with a VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation, and the other uses VGG16 instead of VGG11 ([arxiv paper](https://arxiv.org/abs/1801.05746)).
The GitHub for the TernausNet models can be found [here](https://github.com/ternaus/TernausNet).

The lab gave us several images that we had to label by hand. As a first step, we only labeled the neurons and trained our models on those images. As a second step, we also labeled the axons and trained our models to segment both the neurons and axons.

## Data and models

Before starting we advise you to work on a particular environment for example with Anaconda (conda create --name ml4science) and to have a version of python above or equal to 3.6. Then all the libraries can be download using: 
```bash
pip install -r requirements.txt
```

The original images and their labels (neurons only and neurons + axons) can be found on this [link](https://drive.google.com/drive/folders/1p-e7g9fbw503xHYjhWuaKCHBirZWyk7E?usp=sharing).

The weights of our models, UNet11 and UNet16 for neurons only and for neurons and axons can be found on this [link](https://drive.google.com/drive/folders/1DofS65A4cjx3uWAY7AP0IBJkSSqHukA8?usp=sharing).

Ideally, you should respect the following structure when downloading the data we provide:

```bash
├── ternausnet
│   ├── checkpoints
│   │   └── unet16.pt
│   │   └── ...
│   ├── data
│   │   ├── labels_unordered
│   │   ├── labels_unordered_axons
│   │   ├── originals_unordered
│   │
│   ├── data_job_axons.sh
│   ├── data_job_binary.sh
│   ├── ...
```

## Pre-processing

The images from the lab came in a folder structured like this : 

```bash
├── originals
│   ├── 200_2
│   ├── 200_4
│   ├── 400_2
│   ├── 400_4
│   ├── empty
```

We labeled the images and saved them in a folder following the same structure called *labels*. We also removed duplicate images to obtain a total of 41 images.
We used the script `data_job_binary.sh` which converts the original images and the labeled images to a train - test - validation split (0.7% - 0.2 % - 0.1 %) of the original images and their masks (neurons only).
To do so it creates two new folders *data/labels_unordered* and *data/originals_unordered* which contain the labeled and original images respectively numbered from 1 to 41. Then it creates the masks from the labels and finaly creates the split by creating three folders *train_data*, *val_data*, *test_data* structured like this:

```bash
├── folder_data
│   ├── images
│   │   └──  image_x.tif
│   │   └── ...    
│   ├── masks
│   │   └── mask_x.jpg
│   │   └── ...
```

Moreover each of the 41 images and their masks of size 1536 x 2048 where divided in 4 images of size 768 x 1024 by cropping the original images and masks.

For the labeled axons, we used the script `data_job_axons.sh` which creates the masks for the axons and adds them to the pre-existing masks of the neurons. We thus have a mask for two classes : the neurons and axons (as well as background). Then it created the same train - test - validation split as before but with one extra folder in each split : *masks_with_axons*.

In the end, to use our models for neuron segmentation only just run `data_job_binary.sh`, and for neuron and axon segmentation run `data_job_axons.sh` as well afterwards.

# Training and validation

To train our model, simply run `train.py`. Here is a list of useful arguments:

- `--model` : the model to use, either UNet11 or UNet16.
- `--num_classes` : the number of classes. For neurons only, use a value of 1. For neurons and axons, use a value of 3.
- `--batch_size` : the batch size, which is 4 by default. For num_classes > 1, it is set to 1.
- `--epochs` : the number of epochs to train the model for.
- `--checkpoint_path` : the path to load the model from. Only needed when resuming training from a checkpoint. The path needs to start by *ternausnet/*.
- `--save_checkpoint_name` : the name with which to save the model at the end of each training epoch, for example *unet16.pt*. It needs to end with either *.pt* or *.pth*


To load our data for training and validation, we use dataloaders that respectivaly perform the following operations:
- Randomly crop the images to 512 x 768, perform a vertical flip with probability 0.5, perform a horizontal flip with probability 0.5, then normalize the image.
- Or, perform a center crop to 512 x 768 and then normalize the image.

After each training epoch, we perform validation on the validation data and record the loss and jaccard on the validation set. Then we save the model checkpoint, the losses so far as well as the jaccard values.

# Evaluation

To evaluate our model, run `test.py`. Just like for training, here is a list of useful arguments :

- `--model` : the model to use, either UNet11 or UNet16.
- `--num_classes` : the number of classes. For neurons only, use a value of 1. For neurons and axons, use a value of 3.
- `--checkpoint_path` : the path to load the model from. The path needs to start by *ternausnet/*.
- `--file_names_test` : the relative path to the images to evaluate on, by default it's *test_data/images*.

For each image, it will predict a mask and overlay it on top of the original image. Then it will compute the Intersection over Union (iou or jaccard) for each mask and class and save the results as a pickle file. You will end up with a folder looking like this :

```bash
├── eval
│   ├── masks
│   │   └──  image_x_mask.jpg
│   │   └── ...    
│   ├── overlays
│   │   └── image_x_overlay.jpg
│   │   └── ...
│   └── jaccard.txt
```

When used to predict the segmentation of neurons only (num_classes = 1), the jaccard.txt file will contain a list of jaccard indexes. When used for neurons and axons (num_classes = 3), it will produce a list of lists of the form \[jaccard_neurons, jaccard_axons\]. It also prints on the console the mean jaccard, the standard deviation and the highest jaccard index for each metric.

To plot the learning curves of the different trained models you can run `ternausnet/plot_result.py` and the plots will be saved automatically in the ternausnet folder. 

# Axons and grid analysis

The files for axon directions and grid analysis are in the form of jupyter notebooks because they are more interactive. Indeed the file are meant to create the different figures of the report and can be used as : 
- `draw_axons.ipynb` : Draw the first principal component as well as its orientation on the axons of an image data.
- `find_most_common_slope.ipynb` : Take all images and output the resulting slope of the grid in those images, if there is indeed a grid in the image.
- `orientation.ipynb` : Take an image, process it to find the grid orientation and shows the image with the lines corresponding to the grid slope and print on the image the slope value.
- `plot_histograms.ipynb` : plot the histograms for the different classes of images and perform the t-test to know whether axons grow differently between plates.
