
import os

import numpy as np
import matplotlib.pyplot as plt
from Crypto.Random import get_random_bytes

import encryption
import util
import task

if __name__ == '__main__':

    sampleimg_dir = 'sample/imgs'
    sampleout_dir = 'sample/encrypt'
    if not os.path.isdir(sampleout_dir): os.mkdir(sampleout_dir)
    for f in os.listdir(sampleimg_dir):
        filepath = os.path.join(sampleimg_dir, f)
        img = plt.imread(filepath)
    
        # generate mask according to mask generation rule
        mask = np.logical_not(task.center_mask(img)[...,0].astype(bool))

        # ru


        # generate key
        plt.imshow(eimg)
        plt.show()


        outpath = os.path.join(sampleout_dir, f)
        plt.imsave(outpath,eimg)
