from PIL import Image, ImageChops

from matplotlib import pyplot as plt
from scipy import ndimage
import numpy as np


def convert_to_rgb(img):
    '''
    Convert an image to RGB if needed
    :param img: Pillow image
    :return:
    '''
    if img.mode != 'RGB':
        background = Image.new('RGBA', img.size)
        background.paste(img)
        background = background.convert('RGB')
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
    # max_size = max(img.size)
    # height, width = img.size
    # if height > width:
    #     pos = int((height - width) / 2)
    #     new_img.paste(img, (0, pos))
    # else:
    #     pos = int((width - height) / 2)
    #     new_img.paste(img, (pos, 0))

    return img

#%%

threshold = 250

img = Image.open("images_source/200767/504735_petrovich_11.jpg")
np_im = np.array(img)
np_im[np_im >= threshold] = 255


img = Image.fromarray(np_im, 'RGB')
img = img.rotate(15)

# img = convert_to_rgb(img)
img = trim(img)
np_im = np.array(img)

alpha = np.zeros(np_im.shape[:-1], dtype=np.uint8)


mask = ((np_im == 255).sum(axis=2))

alpha[mask < 3] = 255

# alpha[]

np_im = np.dstack((np_im, alpha))

img = Image.fromarray(np_im, 'RGBA')

img.save("test.png")





# img = np
# plt.imshow(img)


# def preprocess_image(self, img):
#     '''
#     Convert image to RGB and trim
#     :param img:
#     :return:
#     '''
#     rgb_img = convert_to_rgb(img)
#     trim_img = trim(rgb_img)
#     return trim_img


# if __name__ == '__main__':
#     img = Image.open("images_source/200767/90026079_lmmarket_02.jpg")
#     Image._show(img)
#     img = convert_to_rgb(img)
#     img = trim(img)
#     Image._show(img)

