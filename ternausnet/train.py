import torch
import argparse
from torch.optim import Adam
from torch.utils.data import DataLoader
from models import UNet16, UNet11
from dataset import CellsDataset, get_file_names
from loss import LossBinary, LossMulti
import albumentations as A
from utils import train, validate, load_ckp, check_crop_size
import os
import numpy as np
import sys
import pickle


model_list = {'UNet11': UNet11,
               'UNet16': UNet16}


def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # Transforms params
    arg('--min_h', default=768, type=int)
    arg('--min_w', default=1024, type=int)
    arg('--crop_h', default=512, type=int)
    arg('--crop_w', default=768, type=int)

    # Model params
    arg('--model', default='UNet16', type=str, choices=model_list.keys())
    arg('--pretrained', default=True, type=bool)
    arg('--num_classes', default=1, type=int)
    arg('--lr', default=0.001, type=float)
    arg('--batch_size', default=4, type=int)
    arg('--epochs', default=1, type=int)

    # Loading params
    arg('--checkpoint_path', default="", type=str)
    arg('--save_checkpoint_name', default='model_checkpoint.pt', type=str)
    arg('--file_names_train', default='train_data/images', type=str)
    arg('--file_names_val', default='val_data/images', type=str)

    args = parser.parse_args()

    if not check_crop_size(args.crop_h, args.crop_w):
        print('Input image sizes should be divisible by 32, but train '
              'crop sizes ({train_crop_height} and {train_crop_width}) '
              'are not.'.format(train_crop_height=args.crop_h, train_crop_width=args.crop_w))
        sys.exit(0)

    p=1

    train_transform = A.Compose([
                A.PadIfNeeded(min_height=args.min_h, min_width=args.min_w, p=1),
                A.RandomCrop(height=args.crop_h, width=args.crop_w, p=1),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(p=1)
                ], p=p)

    val_transform = A.Compose([
                A.PadIfNeeded(min_height=args.min_h, min_width=args.min_w, p=1),
                A.CenterCrop(height=args.crop_h, width=args.crop_w, p=1),
                A.Normalize(p=1)
                ], p=p)
                

    file_names_train = get_file_names(args.file_names_train)
    file_names_val = get_file_names(args.file_names_val)

    # Define data loaders
    train_dataset = CellsDataset(file_names_train, transform=train_transform, num_classes=args.num_classes)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = CellsDataset(file_names_val, transform=val_transform, num_classes=args.num_classes)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size)

    model_name = model_list[args.model]
    model = model_name(num_classes=args.num_classes, pretrained=args.pretrained)

    optimizer = Adam(model.parameters(), lr=args.lr)

    if args.num_classes > 1:

        criterion = LossMulti(num_classes=args.num_classes)

    else:
        criterion = LossBinary()

    start_epoch = 1
    train_metrics = []
    val_metrics = []

    # Create folders if needed
    metrics_base_path = os.path.join(os.pardir, 'ternausnet/metrics')
    checkpoints_folder_path = os.path.join(os.pardir, 'ternausnet/checkpoints/')
    save_path = os.path.join(os.pardir, 'ternausnet/checkpoints/' + args.save_checkpoint_name)


    if not os.path.exists(metrics_base_path):
        os.makedirs(metrics_base_path)

    if not os.path.exists(checkpoints_folder_path):
        os.makedirs(checkpoints_folder_path)

    # Load the model from checkpoint if needed
    if args.checkpoint_path != "":

        model, optimizer, start_epoch = load_ckp(os.path.join(os.pardir, args.checkpoint_path), model, optimizer)

        epochs = start_epoch + args.epochs - 1

        with open("metrics/train_metrics.txt", "rb") as fp:
            train_metrics = pickle.load(fp)

        with open("metrics/val_metrics.txt", "rb") as fp:
            val_metrics = pickle.load(fp)
    
    else:
        epochs = args.epochs

    
    # Start training
    for epoch in range(start_epoch, epochs + 1):

        l_train, j_train = train(train_loader, model, criterion, optimizer, epoch)
        l_val, j_val = validate(val_loader, model, criterion, epoch, args.num_classes)
        
        train_metrics.append([np.mean(l_train), np.mean(j_train)])
        val_metrics.append([np.mean(l_val), np.mean(j_val)])

        # Save everything
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
        
        with open("metrics/train_metrics.txt", "wb") as fp:
            pickle.dump(train_metrics, fp)

        with open("metrics/val_metrics.txt", "wb") as fp:
            pickle.dump(val_metrics, fp)


if __name__ == '__main__':
    main()