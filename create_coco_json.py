from PIL import Image # (pip install Pillow)
import os
import datetime
import json
from pycococreatortools import pycococreatortools
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon

def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]
            #if pixel != big_cells and pixel != small_cells:
            #    pixel = (0,0,0)
            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)
    return sub_masks

def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    sub_mask = np.array(sub_mask)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')
    #print(contours)
    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(0.01, preserve_topology=False)
        polygons.append(poly)
        #print(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

"""
Output format must be in COCO Object Detection format :
"""
INFO = {
    "description": "Empty",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'small_cell',
    },
    {
        'id': 2,
        'name': 'big_cell',
    },
]

coco_output = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],
    "annotations": []
}


IMAGE_DIR = "/home/james/LIVECell/images/data/empty"
small_cells_s = (0, 179, 205)
small_cells_b = (77, 256, 256)
big_cells_s = (205 ,1, 205)
big_cells_b = (256, 101, 256)

mask_images = []

small_cell_id, big_cell_id = [1, 2]
category_ids = {
    '(28, 230, 255)': small_cell_id,
    '(255, 52, 255)': big_cell_id,
}

image_id = 1
is_crowd = 0

for root, _, files in os.walk(IMAGE_DIR):
    # go through each image
    for image_filename in files:
        image = Image.open(os.path.join(IMAGE_DIR, image_filename))
        pixdata = image.load()
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                if (big_cells_s[0]<=pixdata[x, y][0]<=big_cells_b[0] and big_cells_s[1]<=pixdata[x, y][1]<=big_cells_b[1] and \
                big_cells_s[2]<=pixdata[x, y][2]<=big_cells_b[2]) :
                    pixdata[x, y] = (255, 52, 255)
                elif (small_cells_s[0]<=pixdata[x, y][0]<=small_cells_b[0] and \
                small_cells_s[1]<=pixdata[x, y][1]<=small_cells_b[1] and small_cells_s[2]<=pixdata[x, y][2]<=small_cells_b[2]):
                    pixdata[x, y] = (28, 230, 255)
                #if pixdata[x, y] != small_cells and pixdata[x, y] != big_cells:
                    #print(pixdata[x,y])
                else :
                    pixdata[x, y] = (0, 0, 0)

        #image.show()

        # These ids will be automatically increased as we go
        annotation_id = 1
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_filename), image.size)
        #print(image_info)
        coco_output["images"].append(image_info)

        # Create the annotations
        annotations = []
        sub_masks = create_sub_masks(image)
        for color, sub_mask in sub_masks.items():
            #print(category_ids)
            category_id = category_ids[color]
            annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
            annotations.append(annotation)
            annotation_id += 1
        image_id += 1
        coco_output["annotations"].append(annotations)

with open(os.path.join(IMAGE_DIR, 'empty.json'), 'w') as outfile:
    json.dump(coco_output, outfile)

"""
#plant_book_mask_image = Image.open('/home/james/LIVECell/images/data/empty/image_labelled_0.jpg')#.convert('RGB')
#plant_book_mask_image = Image.open('/home/james/Downloads/plant_book_mask.png').convert('RGB')
#bottle_book_mask_image = Image.open('/home/james/Downloads/bottle_book_mask.png').convert('RGB')

#plant_book_mask_image = np.asarray(plant_book_mask_image.getdata())
#print(np.where(np.all(plant_book_mask_image==[2,2,2], axis=1)))
pixdata = plant_book_mask_image.load()

#print(plant_book_mask_image[0,0])

small_cells_s = (0, 179, 205)
small_cells_b = (77, 256, 256)
big_cells_s = (205, 1, 205)
big_cells_b = (256, 101, 256)

#plant_book_mask_image.show()

for y in range(plant_book_mask_image.size[1]):
    for x in range(plant_book_mask_image.size[0]):
        if (big_cells_s[0]<=pixdata[x, y][0]<=big_cells_b[0] and big_cells_s[1]<=pixdata[x, y][1]<=big_cells_b[1] and \
        big_cells_s[2]<=pixdata[x, y][2]<=big_cells_b[2]) :
            pixdata[x, y] = (255,52,255)
        elif (small_cells_s[0]<=pixdata[x, y][0]<=small_cells_b[0] and \
        small_cells_s[1]<=pixdata[x, y][1]<=small_cells_b[1] and small_cells_s[2]<=pixdata[x, y][2]<=small_cells_b[2]):
            pixdata[x, y] = (28,230,255)
        #if pixdata[x, y] != small_cells and pixdata[x, y] != big_cells:
            #print(pixdata[x,y])
        else :
            pixdata[x, y] = (0,0,0)

plant_book_mask_image.show()
#print(np.where(np.all(plant_book_mask_image==[2,2,2], axis=1)))


mask_images = [plant_book_mask_image, bottle_book_mask_image]

# Define which colors match which categories in the images
houseplant_id, book_id, bottle_id, lamp_id = [1, 2, 3, 4]
category_ids = {
    1: {
        '(28,230,255)': small_cells,
        '(255,52,255)': big_cells,
    },
    2: {
        '(255, 255, 0)': bottle_id,
        '(255, 0, 128)': book_id,
        '(255, 100, 0)': lamp_id,
    }
}

is_crowd = 0

# These ids will be automatically increased as we go
annotation_id = 1
image_id = 1

#Add images
from pycococreatortools import pycococreatortools
IMAGE_DIR = "/home/james/LIVECell/images/data/empty"
for root, _, files in os.walk(IMAGE_DIR):
    #image_files = filter_for_jpeg(root, files)

    # go through each image
    for image_filename in files:
        image = Image.open(os.path.join(IMAGE_DIR, image_filename))
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_filename), image.size)
        print(image_info)
        coco_output["images"].append(image_info)

# Create the annotations
annotations = []
for mask_image in mask_images:
    sub_masks = create_sub_masks(mask_image)
    for color, sub_mask in sub_masks.items():
        #print(color)
        category_id = category_ids[image_id][color]
        annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
        annotations.append(annotation)
        annotation_id += 1
    image_id += 1

#print(json.dumps(annotations))
"""
