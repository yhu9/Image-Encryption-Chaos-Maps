import os

import numpy as np
import matplotlib.pyplot as plt
from Crypto.Random import get_random_bytes

import encryption
import dataloader
import task
from util import fix_mask

# test des encryption/decryption
def test_des():
    key = get_random_bytes(8)
    mode = 'cbc'
    DES = encryption.DESimg(key,mode=mode)

    iv = get_random_bytes(32) if mode == 'cbc' else get_random_bytes(0)
    img = dataloader.getimg('sample/imgs/182839.jpg')
    h,w,d = img.shape

    cipher_img = DES.encrypt(img).reshape(h,w,d)
    decrypt_img = DES.decrypt(cipher_img).reshape(h,w,d)
    showimgs([img,cipher_img, decrypt_img])

# test aes encryption, decryption
def test_aes():
    key = get_random_bytes(32)
    mode = 'ecb'
    AES = encryption.AESimg(key,mode=mode)

    iv = get_random_bytes(32) if mode == 'cbc' else get_random_bytes(0)
    img = dataloader.getimg('sample/imgs/182839.jpg')
    h,w,d = img.shape

    cipher_img = AES.encrypt(img).reshape(h,w,d)
    decrypt_img = AES.decrypt(cipher_img).reshape(h,w,d)
    
    showimgs([img,cipher_img, decrypt_img])

# test rsa encryption, decryption
def test_rsa():
    # test rsa encryption/decryption
    RSA = encryption.RSAimg(initkey=False)
    img = dataloader.getimg('sample/imgs/182839.jpg')
    cipher_img = RSA.encrypt(img)
    decrypt_img = RSA.decrypt(cipher_img)
    
    showimgs([img,cipher_img, decrypt_img])
    
# check if aes works on all masks.
def test_aes_mask():
    
    img = dataloader.getimg('sample/imgs/182839.jpg')

    m1 = np.logical_not(task.random_irregular_mask(img)[...,0].astype(bool))
    m2 = np.logical_not(task.random_regular_mask(img)[...,0].astype(bool))
    m3 = np.logical_not(task.center_mask(img)[...,0].astype(bool))

    AES = encryption.AESimg(get_random_bytes(32),mode='ecb')

    m1 = fix_mask(m1,16)
    m2 = fix_mask(m2,16)
    m3 = fix_mask(m3,16)

    cdata = AES.encrypt(img[m1])
    ddata = AES.decrypt(cdata)
    cimg = img.copy()
    cimg[m1] = cdata.reshape(-1,3)
    dimg = cimg.copy()
    dimg[m1] = ddata.reshape(-1,3)
    f0 = os.path.join('sample/original.png')
    f1 = os.path.join('sample/aes_encrypt_m1.png')
    f2 = os.path.join('sample/aes_decrypt_m1.png')
    showimgs([img,cimg, dimg],show=True, save=True, imgnames=[f0,f1,f2])

    
    cdata = AES.encrypt(img[m2])
    ddata = AES.decrypt(cdata)
    cimg = img.copy()
    cimg[m2] = cdata.reshape(-1,3)
    dimg = cimg.copy()
    dimg[m2] = ddata.reshape(-1,3)
    f0 = os.path.join('sample/original.png')
    f1 = os.path.join('sample/aes_encrypt_m2.png')
    f2 = os.path.join('sample/aes_decrypt_m2.png')
    showimgs([img,cimg, dimg],show=True, save=True, imgnames=[f0,f1,f2])
    
    cdata = AES.encrypt(img[m3])
    ddata = AES.decrypt(cdata)
    cimg = img.copy()
    cimg[m3] = cdata.reshape(-1,3)
    dimg = cimg.copy()
    dimg[m3] = ddata.reshape(-1,3)
    f0 = os.path.join('sample/original.png')
    f1 = os.path.join('sample/aes_encrypt_m3.png')
    f2 = os.path.join('sample/aes_decrypt_m3.png')
    showimgs([img,cimg, dimg],show=True, save=True, imgnames=[f0,f1,f2])

def showimgs(imgs, show=True, save=False, imgnames=[]):
    for i in range(len(imgs)):
        img = imgs[i]
        plt.imshow(img)
        if show: plt.show()
        if save: 
            fname = imgnames[i]
            plt.imsave(fname, img)

# run des on all masks
def test_des_mask():

    return 

# unit test
if __name__ == '__main__':
    print('unit test')

    test_aes_mask()
    # test_des()
    # test_aes()
    # test_rsa()
    