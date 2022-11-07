
import os

import numpy as np
import matplotlib.pyplot as plt
from Crypto.Random import get_random_bytes

import encryption
import util

if __name__ == '__main__':

    sampleimg_dir = 'sample/imgs'
    sampleout_dir = 'sample/encrypt'
    if not os.path.isdir(sampleout_dir): os.mkdir(sampleout_dir)
    for f in os.listdir(sampleimg_dir):
        filepath = os.path.join(sampleimg_dir, f)
        img = plt.imread(filepath)
    
        loc = (200,200)
        size= 720
        key = get_random_bytes(8)
        mode = 'ecb'
        aes_cipher = encryption.DESimg(key,mode=mode)

        eimg = util.encrypt_image(img, aes_cipher, loc, size)

        # plt.imshow(eimg)
        # plt.show()


        outpath = os.path.join(sampleout_dir, f)
        plt.imsave(outpath,eimg)
