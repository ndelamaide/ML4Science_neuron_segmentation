import cv2
import random
import os
import shutil
import argparse


""" For each original image and its mask, divides the image in 4. 
    Then it creates a train-validation-test split of the data and creates the corresponding folders with the images.
    Each folder contains a folder with the original images and the other with the masks. """
    
def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--num_classes', default=1, type=int)

    args = parser.parse_args()

    originals_base_path = os.path.join(os.pardir, 'ternausnet/data/originals_unordered')

    if args.num_classes > 1:
        masks_base_path = os.path.join(os.pardir, 'ternausnet/data/masks_with_axons/')
    else:
        masks_base_path = os.path.join(os.pardir, 'ternausnet/data/masks/')

    train_save_path = os.path.join(os.pardir, 'ternausnet/train_data/')
    val_save_path = os.path.join(os.pardir, 'ternausnet/val_data/')
    test_save_path = os.path.join(os.pardir, 'ternausnet/test_data/')

    temp_save_path = os.path.join(os.pardir, 'ternausnet/temp/')

    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
        os.makedirs(train_save_path+"images")
        os.makedirs(train_save_path+"masks")

    if not os.path.exists(val_save_path):
        os.makedirs(val_save_path)
        os.makedirs(val_save_path+"images")
        os.makedirs(val_save_path+"masks")

    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
        os.makedirs(test_save_path+"images")
        os.makedirs(test_save_path+"masks")

    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)

    # The other folders already exist
    if args.num_classes > 1:
        os.makedirs(train_save_path+"masks_with_axons")
        os.makedirs(val_save_path+"masks_with_axons")
        os.makedirs(test_save_path+"masks_with_axons")


    print("Dividing images in 4")

    num = 1
    for root, dirs, files in os.walk(originals_base_path):
        for file in files:
            if file.endswith(".tif"):

                images = []
                masks = []

                image = cv2.imread(os.path.join(root, file), cv2.IMREAD_UNCHANGED)
                mask = cv2.imread(os.path.join(masks_base_path, file.replace("image", "mask").replace(".tif", ".jpg")), cv2.IMREAD_UNCHANGED)
                
                # divide image + label in 4
                h, w = image.shape
                new_h = h // 2
                new_w = w // 2

                # top left
                images.append(image[:new_h, :new_w])
                masks.append(mask[:new_h, :new_w])

                # top right
                images.append(image[:new_h, new_w:])
                masks.append(mask[:new_h, new_w:])

                # bottom left
                images.append(image[new_h:, :new_w])
                masks.append(mask[new_h:, :new_w])

                # bottom right
                images.append(image[new_h:, new_w:])
                masks.append(mask[new_h:, new_w:])

                for i in range(len(images)):
                    
                    cv2.imwrite(os.path.join(temp_save_path, "image_" +str(num) + ".tif"), images[i])
                    cv2.imwrite(os.path.join(temp_save_path, "mask_" +str(num) + ".jpg"), masks[i])
                    num += 1


    path_to_images = []

    for root, dirs, files in os.walk(temp_save_path):
        for file in files:
            if file.endswith(".tif"):
                path_to_images.append(os.path.join(root, file))

    random.seed(42)
    random.shuffle(path_to_images)

    num_images = len(path_to_images)

    train_num = round(0.7 * num_images)
    test_num = train_num + round(0.2 * num_images)

    print("Creating split")

    for i in range(len(path_to_images)):

        if i < train_num - 1:

            if args.num_classes > 1:
                save_path = train_save_path+"masks_with_axons"
            else : 
                save_path = train_save_path+"masks"
                shutil.move(path_to_images[i], train_save_path+"images") # Only move images for num_classes = 1 because for num_classes = 2 (axons) they already exist

            shutil.move(path_to_images[i].replace("image", "mask").replace(".tif", ".jpg"), save_path)
        
        elif (i >= train_num) & (i < test_num -1):

            if args.num_classes > 1:
                save_path = test_save_path+"masks_with_axons"
            else : 
                save_path = test_save_path+"masks"
                shutil.move(path_to_images[i], test_save_path+"images") 

            shutil.move(path_to_images[i].replace("image", "mask").replace(".tif", ".jpg"), save_path)
        
        else:

            if args.num_classes > 1:
                save_path = val_save_path+"masks_with_axons"
            else : 
                save_path = val_save_path+"masks"
                shutil.move(path_to_images[i], val_save_path+"images")

            shutil.move(path_to_images[i].replace("image", "mask").replace(".tif", ".jpg"), save_path)


    shutil.rmtree(temp_save_path)


if __name__ == '__main__':
    main()