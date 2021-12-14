from validation import predict
from dataset import get_file_names
import pickle
import os
from utils import load_ckp
from albumentations import Compose, Normalize
from ternausnet.models import UNet16
from torch.optim import Adam
import argparse

def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # Loading params
    arg('--checkpoint_path', default="ternausnet/checkpoints/model_checkpoint.pt", type=str)
    arg('--file_names_test', default='test_data/images', type=str)
    arg('--save_path', default='ternausnet/eval/', type=str)

    args = parser.parse_args()

    model = UNet16(pretrained=True)
    optimizer = Adam(model.parameters())

    model, optimizer, start_epoch = load_ckp(os.path.join(os.pardir,args.checkpoint_path), model, optimizer)
    model.eval()

    p=1
    img_transform = Compose([Normalize(p=1)], p=p)

    file_names = get_file_names(args.file_names_test)
    to_path = os.path.join(os.pardir, args.save_path)

    if not os.path.exists(to_path):
        os.makedirs(to_path)

    predict(model, file_names, to_path, img_transform)

if __name__ == '__main__':
    main()