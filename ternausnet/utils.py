import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def train(train_loader, model, criterion, optimizer, epoch):
    
    model.train()

    losses = []
    jaccard = []

    stream = tqdm(train_loader)

    for i, (images, target) in enumerate(stream, start=1):
        
        output = model(images)
        loss = criterion(output, target)

        with torch.no_grad():
            jaccard_ = get_jaccard(target, (output > 0).float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train. | Loss: {loss}  | Jaccard: {jacc}".format(epoch=epoch, loss=loss.item(), jacc=jaccard_)
        )

        losses.append(loss.item())
        jaccard.append(jaccard_)
    
    return losses, jaccard


def validate(val_loader, model, criterion, epoch):

    with torch.no_grad():

        model.eval()

        losses = []
        jaccard = []

        stream = tqdm(val_loader)
    
        for i, (images, target) in enumerate(stream, start=1):

            output = model(images)
            loss = criterion(output, target)
            jaccard_ = get_jaccard(target, (output > 0).float())

            stream.set_description(
                "Epoch: {epoch}. Validation. | Loss: {loss}  | Jaccard: {jacc}".format(epoch=epoch, loss=loss.item(), jacc=jaccard_)
            )

            losses.append(loss.item())
            jaccard.append(jaccard_)
    
    return losses, jaccard


def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum()
    union = y_true.sum() + y_pred.sum() - intersection

    return (intersection + epsilon) / (union + epsilon)

def check_crop_size(image_height, image_width):
    """Checks if image size divisible by 32.
    Args:
        image_height:
        image_width:
    Returns:
        True if both height and width divisible by 32 and False otherwise.
    """
    return image_height % 32 == 0 and image_width % 32 == 0

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['model_state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # return model, optimizer, epoch value
    return model, optimizer, checkpoint['epoch']