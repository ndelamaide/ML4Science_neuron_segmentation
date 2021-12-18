from validation import predict
from dataset import get_file_names
import pickle
import os
from utils import load_ckp
from albumentations import Compose, Normalize
from models import UNet16, UNet11
from torch.optim import Adam
import argparse


model_list = {'UNet11': UNet11,
               'UNet16': UNet16}


def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # Loading params
    arg('--checkpoint_path', default="ternausnet/checkpoints/model_checkpoint.pt", type=str)
    arg('--file_names_test', default='test_data/images', type=str)
    arg('--save_path', default='ternausnet/eval/', type=str)

    # Model params
    arg('--model', default='UNet16', type=str, choices=model_list.keys())
    arg('--num_classes', default=1, type=int)
    
    args = parser.parse_args()

    p=1
    img_transform = Compose([Normalize(p=1)], p=p)

    file_names = get_file_names(args.file_names_test)
    to_path = os.path.join(os.pardir, args.save_path)

    if not os.path.exists(to_path):
        os.makedirs(to_path)

    model_name = model_list[args.model]
    model = model_name(num_classes=args.num_classes, pretrained=args.pretrained)
    optimizer = Adam(model.parameters())

    model, optimizer, start_epoch = load_ckp(os.path.join(os.pardir,args.checkpoint_path), model, optimizer)
    model.eval()

    predict(model, file_names, to_path, img_transform)

    # After prediction run evaluation

if __name__ == '__main__':
    main()