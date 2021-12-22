# LIVECell adaptation

The goal of the project was to perform image segmentation on images of neuronal cells growing on a diamond.

This is an adapted training procedure for transfer learning from LIVECell ([nature paper](https://www.nature.com/articles/s41592-021-01249-6)), with code that can be found [here] (https://github.com/sartorius-research/LIVECell). 

As a first step yxou should clone LIVECell repository locally, as well as centermask2, following this structure when downloading the data we provide:

```bash
├── LIVECell
│   ├── code
│   │   └── coco_evaluation.py
│   │   └── preprocessing.py
│   ├── images
│   │   ├── labelled_data
│   │   ├── sliced
│   │   ├── unlabelled_data
│   ├── anchor_free
│   │   ├── centermask2
│   │   └── train_net.py
│   │   └── livecell_config.yaml
│   ├── ...
|   ...
```

LIVECell uses a pretrained model named centermask2 that can be found [here](https://github.com/youngwanLEE/centermask2) and that is based on [detectron2] (https://github.com/facebookresearch/detectron2). 

Before starting we advise you to work on a particular environment for example with Anaconda (conda create --name livecell) and to have a version of python above or equal to 3.6. Then all the libraries can be download using: 
```bash
pip install -r requirements.txt
```

The lab gave us several images that we had to label by hand. As a first step, we only labeled the neurons and trained our models on those images. As a second step, we also labeled the axons and trained our models to segment both the neurons and axons.

The original images and their labels (neurons only and neurons + axons) can be found on this [link](https://drive.google.com/drive/folders/1p-e7g9fbw503xHYjhWuaKCHBirZWyk7E?usp=sharing).

The labels have to be transformed into COCO dataset for detectron2 to train and evaluate the network. For this, we first have to slice the images using the file `slice.py` to obtain images with good dimensions. Then, we can augment the dataset using the script `data_augmentation.py`. To create the COCO annotations we can then run `create_coco_json.py`. `coco_show.ipynb` and `show_coco_image.py` are just helpers to visualize the resulting annotated images and `create_folders.py? allows to separate the data and annotations following the project's structure. 

To run the model, we first have to modify the config file named `livecell_config.yaml`and sprcify the number_of_workers, the train and test set. Then we can run the following command to train the network : 
```bash
python train_net.py --config-file livecell_config.yaml MODEL.WEIGHTS http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_free/ALL/LIVECell_anchor_free_model.pth MODEL.DEVICE cpu
```
We specified the MODEL.DEVICE cpu because we had none GPU available for training. 

Then the performance can be test by adding `--eval-only`in the command bash. 
