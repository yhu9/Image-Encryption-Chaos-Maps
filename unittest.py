

import encryption
import dataloader

import matplotlib.pyplot as plt
from Crypto.Random import get_random_bytes

# test des encryption/decryption
def test_des():
    key = get_random_bytes(8)
    mode = 'cbc'
    DES = encryption.DESimg(key,mode=mode)

    iv = get_random_bytes(32) if mode == 'cbc' else get_random_bytes(0)
    img = dataloader.getimg('sample/imgs/182839.jpg')
    cipher_img = DES.encrypt(img)
    plt.imshow(img)
    plt.show()
    plt.imshow(cipher_img)
    plt.show()
    decrypt_img = DES.decrypt(cipher_img)
    plt.imshow(decrypt_img)
    plt.show()

# test aes encryption, decryption
def test_aes():
    key = get_random_bytes(32)
    mode = 'ecb'
    AES = encryption.AESimg(key,mode=mode)

    iv = get_random_bytes(32) if mode == 'cbc' else get_random_bytes(0)
    img = dataloader.getimg('sample/imgs/182839.jpg')
    cipher_img = AES.encrypt(img)
    plt.imshow(img)
    plt.show()
    plt.imshow(cipher_img)
    plt.show()
    decrypt_img = AES.decrypt(cipher_img)
    plt.imshow(decrypt_img)
    plt.show()

# test rsa encryption, decryption
def test_rsa():
    # test rsa encryption/decryption
    RSA = encryption.RSAimg(initkey=False)
    img = dataloader.getimg('sample/imgs/182839.jpg')
    cipher_img = RSA.encrypt(img)
    plt.imshow(img)
    plt.show()
    plt.imshow(cipher_img)
    plt.show()
    decrypt_img = RSA.decrypt(cipher_img)
    plt.imshow(decrypt_img)
    plt.show()

# unit test
if __name__ == '__main__':
    print('unit test')

    # test_des()
    # test_aes()
    test_rsa()
    