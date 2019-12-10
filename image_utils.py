
import os
from PIL import Image, ImageChops
from torchvision import transforms
import random

from matplotlib import pyplot as plt
# from scipy import ndimage
import numpy as np


def convert_to_rgb(img):
    '''
    Convert an image to RGB if needed
    :param img: Pillow image
    :return:
    '''
    if img.mode != 'RGB':
        img.convert('RGBA')
        background = Image.new('RGB', img.size, "WHITE")
        background.paste(img, (0, 0), img)
        # background = background.convert('RGB')
        return background
    return img


def trim(img):
    '''
    Trim image white border and make it square
    with image in the middle
    :param img: Pillow image
    :return:
    '''

    bg = Image.new('RGB', img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    img = img.crop(bbox)
    return img


#%%

angle = 25

obj_transforms = transforms.Compose([
    # transforms.Resize(450),
    transforms.RandomRotation(degrees=angle, expand=True),
    transforms.RandomPerspective(  # Performs Perspective transformation randomly with a given probability
        distortion_scale=0.45,      # controls the degree of distortion and ranges from 0 to 1
        p=1                        # probability of the image being perspectively transformed
    ),
    transforms.RandomHorizontalFlip(),
    # transforms.CenterCrop(350),
    transforms.ColorJitter(
        brightness=0.2,  # 0.3
        contrast=0.2,  #
        saturation=0.2
    )
])

bg_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=angle, expand=True),
    transforms.RandomPerspective(  # Performs Perspective transformation randomly with a given probability
        distortion_scale=0.5,       # controls the degree of distortion and ranges from 0 to 1
        p=0.8                       # probability of the image being perspectively transformed
    ),
])


def remove_bg(input_img):
    threshold = 253

    input_img = convert_to_rgb(input_img)

    np_img = np.array(input_img)
    np_img[np_img >= threshold] = 255

    input_img = Image.fromarray(np_img, 'RGB')

    np_img = np.array(input_img)

    alpha = np.zeros(np_img.shape[:-1], dtype=np.uint8)
    mask = ((np_img == 255).sum(axis=2))
    alpha[mask < 3] = 255
    np_img = np.dstack((np_img, alpha))

    return Image.fromarray(np_img, 'RGBA')


def transform(input_img):
    input_img = convert_to_rgb(input_img)

    input_img = remove_bg(input_img)
    input_img = obj_transforms(input_img)

    input_img = convert_to_rgb(input_img)
    input_img = trim(input_img)
    input_img = remove_bg(input_img)

    return input_img


img = Image.open("images_source/200767/504735_petrovich_11.jpg")

bg = Image.open("backgrounds/D2115_158_005_1200.jpg")

for i in range(30):

    test_img = transform(img)
    # test_img = obj_transforms(img)

    test_img.save(os.path.join("test", f"test_{i}.png"))
