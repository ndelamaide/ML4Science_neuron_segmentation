import numpy as np
from torch.utils.data import DataLoader
from dataset import CellsDataset
import torch
from tqdm import tqdm
import cv2



def predict(model, from_file_names, to_path, img_transform):
    loader = DataLoader(
        dataset=CellsDataset(from_file_names, transform=img_transform, mode='predict'),
        shuffle=False
    )

    num = 0
    with torch.no_grad():
        for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):

            outputs = model(inputs)

            for i, image_name in enumerate(paths):

                factor = 255
                t_mask = torch.sigmoid(outputs[i, 0]).data.cpu().numpy()
                im_mask = (t_mask * factor).astype(np.uint8)
                t_mask[t_mask >= 0.3] = 1
                t_mask[t_mask < 0.3] = 0
                t_mask = (t_mask * factor).astype(np.uint8)
                
                name = image_name.split("/")[-1][:-4]

                cv2.imwrite(to_path + name + '.png', im_mask)
                cv2.imwrite(to_path + name + "_mask"+ '.png', t_mask)
                num += 1