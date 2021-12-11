import cv2
import os

IMAGE_DIR = "/home/james/LIVECell/images/labelled_data/"
#IMAGE_DIR = "/home/james/LIVECell/images/labelled_data/"
OUTPUT_DIR = "/home/james/LIVECell/images/sliced/labelled/"
#OUTPUT_DIR = "/home/james/LIVECell/images/sliced/labelled/"


for folder in os.listdir(IMAGE_DIR):
    # go through each image
    print(folder)
    #for f in os.listdir(os.path.join(OUTPUT_DIR, folder)):
    #    os.remove(os.path.join(IMAGE_DIR, folder, f))
    for root, dirs, images in os.walk(os.path.join(IMAGE_DIR, folder)):
        for image in images :
            if os.path.splitext(image)[1].lower() in ('.jpg', '.tif'):
                img = cv2.imread(os.path.join(root, image))
                diff = [int((520*3-img.shape[0])/2), int((704*3-img.shape[1])/2)]
                y1 = 520-diff[0]
                y2 = 1536-520
                x1 = 704-diff[1]
                x2 = 2048-704
                crop_imgs = [img[0:520,0:704,:], img[y1:y1+520,0:704,:], img[y2:1536,0:704,:],\
                 img[0:520,x1:x1+704,:], img[y1:y1+520,x1:x1+704,:], img[y2:1536,x1:x1+704,:],\
                 img[0:520,x2:2048,:], img[y1:y1+520,x2:2048,:], img[y2:1536,x2:2048,:]]
                for i in range(len(crop_imgs)) :
                    #print(os.path.join(OUTPUT_DIR, folder, os.path.splitext(image)[0]+"_"+str(i)+".jpg"))
                    #print(folder)
                    #print(image)
                    cv2.imwrite(os.path.join(OUTPUT_DIR, folder, os.path.splitext(image)[0]+"_"+str(i)+".jpg"), crop_imgs[i])
