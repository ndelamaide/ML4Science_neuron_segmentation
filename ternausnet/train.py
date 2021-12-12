import torch
import argparse
from torch.optim import Adam
from torch.utils.data import DataLoader
from ternausnet.models import UNet16
from dataset import CellsDataset, get_file_names
from loss import LossBinary
import albumentations as A
from utils import train, validate, load_ckp, check_crop_size
import os
import numpy as np
import sys
import pickle


def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # Transforms params
    arg('--min_h', default=768, type=int)
    arg('--min_w', default=1024, type=int)
    arg('--crop_h', default=512, type=int)
    arg('--crop_w', default=768, type=int)

    # Model params
    arg('--lr', default=0.001, type=float)
    arg('--batch_size', default=4, type=int)
    arg('--epochs', default=1, type=int)

    # Loading params
    arg('--checkpoint_path', default="", type=str)
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
    train_dataset = CellsDataset(file_names_train, transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = CellsDataset(file_names_val, transform=val_transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size)


    model = UNet16(pretrained=True)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = LossBinary()
    start_epoch = 1

    train_metrics = []

    val_metrics = []

    # Create folders if needed
    metrics_base_path = os.path.join(os.pardir, 'ternausnet/metrics')

    if not os.path.exists(metrics_base_path):
        os.makedirs(metrics_base_path)
    
    models_base_path = os.path.join(os.pardir, 'ternausnet/models')

    if not os.path.exists(models_base_path):
        os.makedirs(models_base_path)

    # Load the model if needed
    if args.checkpoint_path != "":
        model, optimizer, start_epoch = load_ckp(os.path.join(os.pardir, args.checkpoint_path), model, optimizer)

        with open("metrics/train_metrics.txt", "rb") as fp:
            train_metrics = pickle.load(fp)

        with open("metrics/val_metrics.txt", "rb") as fp:
            val_metrics = pickle.load(fp)
    

    for epoch in range(start_epoch, args.epochs + 1):
        l_train, j_train = train(train_loader, model, criterion, optimizer, epoch)
        l_val, j_val = validate(val_loader, model, criterion, epoch)
        
        train_metrics.append([np.mean(l_train), np.mean(j_train)])
        val_metrics.append([np.mean(l_val), np.mean(j_val)])

        # Save everything
        save_path = os.path.join(os.pardir, 'ternausnet/models/model_checkpoint.pt')

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