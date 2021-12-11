import  os
import shutil

FILEDIR = "/home/james/LIVECell/images/sliced/unlabelled"
for root, _, images in os.walk(FILEDIR):
    for image in images:
        os.mkdir(os.path.join(root, os.path.splitext(image)[0].lower()))
        shutil.move(os.path.join(root, image),os.path.join(root, os.path.splitext(image)[0].lower(), image))
