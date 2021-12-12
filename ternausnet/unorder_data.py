import os
import cv2


""" Unorders the data.
    This supposes that the original images and their labels are in separate folders
    containing sub-folders 200_2, 200_4 etc..
    The outputs are folders containing the original images and the labels named like the following:
    image_x.tif and label_x.jpg were x is an integer. An original image and its label have the same number. """

def main():

    root_path = os.path.join(os.pardir, "ternausnet/data/labels")

    save_path_labels = os.path.join(os.pardir, "ternausnet/data/labels" + "_unordered/")
    save_path_originals = os.path.join(os.pardir, "ternausnet/data/originals" + "_unordered/")

    if not os.path.exists(save_path_labels):
        os.makedirs(save_path_labels)

    if not os.path.exists(save_path_originals):
        os.makedirs(save_path_originals)

    img_num = 1

    for root, dirs, files in os.walk(root_path, topdown=True):
        for file in files:
            if file.endswith(".jpg"):

                # label
                image = cv2.imread(os.path.join(root, file), cv2.IMREAD_UNCHANGED)
                cv2.imwrite(os.path.join(save_path_labels,  "label_" + str(img_num) + ".jpg"), image)

                num = file.split("_")[-1][:-4]

                #original
                image = cv2.imread(os.path.join(root.replace("labels", "originals"), str(num)+".tif"), cv2.IMREAD_UNCHANGED)
                cv2.imwrite(os.path.join(save_path_originals,  "image_" + str(img_num) + ".tif"), image)
                img_num += 1

if __name__ == '__main__':
    main()