from random import randint
import numpy as np
import cv2
from PIL import Image
import random

###################################################################
# random mask generation
###################################################################
def random_irregular_mask(img):
    """Generates a random irregular mask with lines, circles and elipses"""
    mask = np.ones_like(img)
    h,w,d = img.shape
    img = np.zeros((h, w, 1), np.uint8)

    # Set size scale
    max_width = 20
    if h < 64 or w < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    number = random.randint(16, 64)
    for _ in range(number):
        model = random.random()
        if model < 0.6:
            # Draw random lines
            x1, x2 = randint(1, h), randint(1, h)
            y1, y2 = randint(1, w), randint(1, w)
            thickness = randint(4, max_width)
            cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        elif model > 0.6 and model < 0.8:
            # Draw random circles
            x1, y1 = randint(1, h), randint(1, w)
            radius = randint(4, max_width)
            cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

        elif model > 0.8:
            # Draw random ellipses
            x1, y1 = randint(1, h), randint(1, w)
            s1, s2 = randint(1, h), randint(1, w)
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(4, max_width)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

    img_mask = img.reshape(w, h)
    # img = Image.fromarray(img*255)

    for j in range(d):
        mask[:,:,j] = img_mask < 1

    return mask


def random_regular_mask(img):
    """Generates a random regular hole"""
    mask = np.ones_like(img)
    h,w,d = img.shape
    N_mask = random.randint(1, 5)
    limx = w - w / (N_mask + 1)
    limy = h - h / (N_mask + 1)
    for _ in range(N_mask):
        x = random.randint(0, int(limx))
        y = random.randint(0, int(limy))
        range_x = x + random.randint(int(w / (N_mask + 7)), int(w - x))
        range_y = y + random.randint(int(h / (N_mask + 7)), int(h - y))
        mask[int(x):int(range_x), int(y):int(range_y)] = 0
    return mask

def center_mask(img):
    """Generates a center hole with 1/4*W and 1/4*H"""
    mask = np.ones_like(img)
    h,w,d = img.shape
    x = int(w / 4)
    y = int(h / 4)
    range_x = int(w * 3 / 4)
    range_y = int(h * 3 / 4)
    mask[y:range_y, x:range_x] = 0

    return mask