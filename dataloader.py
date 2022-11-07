
import os

import util

import numpy as np
import matplotlib.pyplot as plt


# read image file and get image
# filename      str         full file path
#OUTPUT:
# img           [h,w,3]     numpy array image
def getimg(filename):
    img = plt.imread(filename)
    if len(img.shape) == 2:
        img = np.stack([img] * 3,axis=-1)
    if img.shape[-1] == 4:
        img = img[...,:3]
    return img

# unit test
if __name__ == '__main__':

    sample_dir = 'sample'
    for f in os.listdir('sample'):
        filename = os.path.join(sample_dir, f)

        img = getimg(filename)

        plt.imshow(img)
        plt.show()

    


