import PIL
import numpy as np
from PIL import Image, ExifTags


def load_image_file(image_path):
    image = PIL.Image.open(image_path)
    if image._getexif() is not None:
        image_orientation = get_image_orientation(image)
        image = transpose_image(image, image_orientation)
    return np.array(image)


def get_image_orientation(image):
    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in image._getexif().items()
        if k in PIL.ExifTags.TAGS
    }
    image_orientation = exif.get("Orientation")
    return image_orientation


def transpose_image(image, image_orientation):
    if image_orientation == 2:
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    if image_orientation == 3:
        image = image.transpose(PIL.Image.ROTATE_180)
    if image_orientation == 4:
        image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    if image_orientation == 5:
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT).transpose(PIL.Image.ROTATE_90)
    if image_orientation == 6:
        image = image.transpose(PIL.Image.ROTATE_270)
    if image_orientation == 7:
        image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM).transpose(PIL.Image.ROTATE_90)
    if image_orientation == 8:
        image = image.transpose(PIL.Image.ROTATE_90)
    return image
