import torch
from tqdm import tqdm
import cv2
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


def validate(val_loader, model, criterion, epoch, num_classes):

    with torch.no_grad():

        model.eval()

        losses = []
        jaccard = []

        if num_classes > 1:
            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.uint32)

        stream = tqdm(val_loader)
    
        for i, (images, target) in enumerate(stream, start=1):

            output = model(images)
            loss = criterion(output, target)

            if num_classes > 1:

                output_classes = output.data.numpy().argmax(axis=1)
                target_classes = target.data.numpy()

                matrix = calculate_confusion_matrix_from_arrays(output_classes, target_classes, num_classes)
                confusion_matrix += matrix

                jaccard_ = np.mean(calculate_iou(confusion_matrix))

            else:

                jaccard_ = get_jaccard(target, (output > 0).float())
                
            
            jaccard.append(jaccard_)

            stream.set_description(
                "Epoch: {epoch}. Validation. | Loss: {loss}  | Jaccard: {jacc}".format(epoch=epoch, loss=loss.item(), jacc=jaccard[-1])
            )

            losses.append(loss.item())     
        
        return losses, jaccard


def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum()
    union = y_true.sum() + y_pred.sum() - intersection

    return (intersection + epsilon) / (union + epsilon)


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    
    replace_indices = np.vstack((ground_truth.flatten(), prediction.flatten())).T
    confusion_matrix, _ = np.histogramdd(replace_indices,bins=(nr_labels, nr_labels),
                                            range=[(0, nr_labels), (0, nr_labels)])

    confusion_matrix = confusion_matrix.astype(np.uint32)

    return confusion_matrix


def calculate_iou(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious


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


def overlay(image, mask, num_classes=1):

    zeros = np.zeros(mask.shape)
    mask_color = np.empty((mask.shape[0], mask.shape[1], 3))

    if num_classes > 1:

        ## TODO: CHECK SIZE OF MASK 
        mask_color[:, :, 0] = mask[mask > 128] # Axons have value > 128
        mask_color[:, :, 1] = mask[mask <= 128]
        mask_color[:, :, 2] = zeros

    else:

        # Turn mask green
        mask_color[:, :, 0] = zeros
        mask_color[:, :, 1] = mask
        mask_color[:, :, 2] = zeros

    img_overlay = cv2.addWeighted(image, 0.8, mask_color.astype(np.uint8), 0.2, 0)

    return img_overlay
    