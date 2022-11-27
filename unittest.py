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

#run arnold on all masks
def test_arnold_mask():
    # img = dataloader.getimg('sample/imgs/189551.jpg')
    # img = dataloader.getimg('sample/orig.png')
    # img = dataloader.getimg('sample/189551.png')
    img = dataloader.getimg('sample/imgs/182839.jpg')


    # m1 = np.logical_not(task.random_irregular_mask(img)[...,0].astype(bool))
    m1 = np.logical_not(task.center_mask(img)[...,0].astype(bool))
    print(m1.shape)
    m1 = fix_mask(m1,16)
    cimg = img.copy()
    print(cimg[m1].shape)
    print(m1.shape)
    cimg[m1] = encryption.ArnoldCatEncryption(cimg[m1], m1, 20)
    dimg  = cimg.copy()
    dimg[m1] = encryption.ArnoldCatDecryption(dimg[m1], m1, 20)

 
    # cimg = img.copy()
    # print(img.shape)
    # cimg = encryption.ArnoldCatEncryption(cimg, m1, 20)
    # dimg  = cimg.copy()
    # dimg = encryption.ArnoldCatDecryption(dimg, m1, 20)

    f0 = os.path.join('sample/original.png')
    f1 = os.path.join('sample/arnold_encrypt_m1.png')
    f2 = os.path.join('sample/arnold_decrypt_m1.png')

    showimgs([img,cimg, dimg],show=True, save=True, imgnames=[f0,f1,f2])

    m1 = fix_mask(m1,16)    
    return

# # run des on all masks
# def test_henon_mask():
#     img = dataloader.getimg('sample/189551.png')
#     # img = dataloader.getimg('sample/imgs/189551.jpg')


#     m1 = np.logical_not(task.center_mask(img)[...,0].astype(bool))
#     m1 = fix_mask(m1,16)
#     for i in img:
#         for j in i:
#             for k in j:
#                 if(k < 0 or k > 1):
#                     print(k)
#     cimg = img.copy()
#     print(img.shape)
#     cimg = encryption.HenonEncryption(cimg, m1, [0.1,0.1])
#     cimg = np.asarray(cimg, dtype=np.float32)
#     print(cimg[0][0])
#     # for i in cimg:
#     #     for j in i:
#     #         for k in j:
#     #             if(k < 0 or k > 1):
#     #                 print(k)
#     dimg  = cimg.copy()
#     dimg = encryption.HenonDecryption(dimg, m1, [0.1,0.1])
#     dimg = np.asarray(dimg, dtype=np.float32)

#     f0 = os.path.join('sample/original.png')
#     f1 = os.path.join('sample/henon_encrypt_m1.png')
#     f2 = os.path.join('sample/henon_decrypt_m1.png')
#     # f3 = os.path.join('sample/arnold_test_m1.png')
#     showimgs([img,cimg, dimg],show=True, save=True, imgnames=[f0,f1,f2])

#     return 

# run des on all masks
def test_henon_mask():
    # img = dataloader.getimg('sample/189551.png')
    # img = dataloader.getimg('sample/imgs/189551.jpg')
    img = dataloader.getimg('sample/imgs/190909.jpg')


    m1 = np.logical_not(task.random_irregular_mask(img)[...,0].astype(bool))
    m2 = np.logical_not(task.random_regular_mask(img)[...,0].astype(bool))
    m3 = np.logical_not(task.center_mask(img)[...,0].astype(bool))
    

    m1 = fix_mask(m1,16)
    m2 = fix_mask(m2,16)
    m3 = fix_mask(m3,16)

    cimg = img.copy()
    masked_cimg = np.array([cimg[m1]], dtype=np.uint8)
    cimg[m1] = encryption.HenonEncryption(masked_cimg, m1, [0.1,0.1])[0]
    cimg = np.asarray(cimg, dtype=np.uint8)
    # for i in cimg:
    #     for j in i:
    #         for k in j:
    #             if(k < 0 or k > 1):
    #                 print(k)
    dimg  = cimg.copy()
    masked_dimg = np.array([dimg[m1]],dtype=np.uint8)
    dimg[m1] = encryption.HenonDecryption(masked_dimg, m1, [0.1,0.1])[0]
    dimg = np.asarray(dimg,dtype=np.uint8)


    f0 = os.path.join('sample/original.png')
    f1 = os.path.join('sample/henon_encrypt_m1.png')
    f2 = os.path.join('sample/henon_decrypt_m1.png')
    # f3 = os.path.join('sample/arnold_test_m1.png')
    showimgs([img,cimg, dimg],show=True, save=True, imgnames=[f0,f1,f2])

    cimg = img.copy()
    masked_cimg = np.array([cimg[m2]])
    cimg[m2] = encryption.HenonEncryption(masked_cimg, m2, [0.1,0.1])[0]
    cimg = np.asarray(cimg,dtype=np.uint8)
    dimg  = cimg.copy()
    masked_dimg = np.array([dimg[m2]])
    dimg[m2] = encryption.HenonDecryption(masked_dimg, m2, [0.1,0.1])[0]
    dimg = np.asarray(dimg,dtype=np.uint8)


    f0 = os.path.join('sample/original.png')
    f1 = os.path.join('sample/henon_encrypt_m2.png')
    f2 = os.path.join('sample/henon_decrypt_m2.png')
    # f3 = os.path.join('sample/arnold_test_m1.png')
    showimgs([img,cimg, dimg],show=True, save=True, imgnames=[f0,f1,f2])

    cimg = img.copy()
    masked_cimg = np.array([cimg[m3]])
    cimg[m3] = encryption.HenonEncryption(masked_cimg, m3, [0.1,0.1])[0]
    cimg = np.asarray(cimg,dtype=np.uint8)
    dimg  = cimg.copy()
    masked_dimg = np.array([dimg[m3]])
    dimg[m3] = encryption.HenonDecryption(masked_dimg, m3, [0.1,0.1])[0]
    dimg = np.asarray(dimg,dtype=np.uint8)


    f0 = os.path.join('sample/original.png')
    f1 = os.path.join('sample/henon_encrypt_m3.png')
    f2 = os.path.join('sample/henon_decrypt_m3.png')
    # f3 = os.path.join('sample/arnold_test_m1.png')
    showimgs([img,cimg, dimg],show=True, save=True, imgnames=[f0,f1,f2])

#     return 

# run des on all masks
def test_des_mask():

    return 

# unit test
if __name__ == '__main__':
    print('unit test')
    # test_arnold_mask()
    test_henon_mask()
    # test_aes_mask()
    # test_des()
    # test_aes()
    # test_rsa()
    