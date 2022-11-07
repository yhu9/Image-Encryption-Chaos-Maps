import os

import util

import numpy as np
import dataloader


# run on sample images
if __name__ == '__main__':
    sample_dir = 'sample'
    for f in os.listdir('sample'):
        filename = os.path.join(sample_dir, f)

        img = dataloader.getimg(filename)
        encry

    print('hello world')