from PIL import Image
import numpy as np
import glob
import os

def merge_images(file1, file2):
    """Merge two images into one, displayed side by side
    :param file1: path to first image file
    :param file2: path to second image file
    :return: the merged Image object
    """
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new('I', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result

l = sorted(glob.glob('/home/george/Documents/ddsm/pics/*/fl*/*.png'))
for left_cc in l[::4]:
    right_cc = left_cc.replace('LEFT_', 'RIGHT_')
    together = left_cc.replace('LEFT_', '')
    folder, filename = os.path.split(together)
    together = os.path.join(os.path.dirname(folder), 'merged',
            os.path.basename(folder).replace('fl', '') + '_' + filename)
    merge_images(left_cc, right_cc).save(together)
