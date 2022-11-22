
import argparse
import os

import matplotlib.pyplot as plt
from Crypto.Random import get_random_bytes
import task

import encryption
import util

if __name__ == '__main__':

    data_dir = 'data/celebA/img_align_celeba'
    sampleout_dir = 'sample/celebA/aes'
    if not os.path.isdir(sampleout_dir): os.mkdir(sampleout_dir)

    img_files = os.listdir(data_dir)
    img_files.sort()
    for f in img_files:
        filepath = os.path.join(data_dir, f)
        img = plt.imread(filepath)

        # generate mask according to mask generation rule
        mask = task.center_mask(img)
    
        loc = (200,200)
        size= 720
        key = get_random_bytes(8)
        mode = 'ecb'
        aes_cipher = encryption.AESimg(key,mode=mode)

        eimg = util.encrypt_image(img, aes_cipher, loc, size)

        # plt.imshow(eimg)
        # plt.show()


        outpath = os.path.join(sampleout_dir, f)
        plt.imsave(outpath,eimg)
