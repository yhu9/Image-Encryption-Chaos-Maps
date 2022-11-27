import math

from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow

from Crypto.Cipher import AES, DES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from math import log

# RSA Encryption on images using Crypto Module
class RSAimg():
    def __init__(self,keysize=4096,initkey=False):
        self.keysize=keysize
        
        # generate private and public key pair for rsa and save in rsakey directory
        if initkey: self.genkey(self.keysize)
        
        self.private_key = RSA.importKey(open('rsakey/privatekey.pem').read())
        self.public_key = RSA.importKey(open('rsakey/publickey.pem').read())
        

        self.private_cipher = PKCS1_OAEP.new(self.private_key)
        self.public_cipher = PKCS1_OAEP.new(self.public_key)

        # for understanding max message length see
        # https://crypto.stackexchange.com/questions/42097/what-is-the-maximum-size-of-the-plaintext-message-for-rsa-oaep
        self.msglen = 256

    # define key and key size
    #
    #INPUT:
    # img           [h,w,3]         numpy array as image
    #OUTPUT:
    # cimg          [h+1,w,3]       cipher image
    def encrypt(self,img,mode='public'):
        h, w, d = img.shape
        byte_img = img.tobytes()
        ciphertext = []
        if mode == 'private':
            for i in range(int(math.ceil(len(byte_img)) / self.msglen)):
                ciphertext.append(self.private_cipher.encrypt(byte_img[int(i*self.msglen):int((i+1)*self.msglen)]))

                # ciphertext.append(pow(int(byte_img[int(i*self.msglen):int((i+1)*self.msglen)], 2), self.private_key.e, self.private_key.n))
            ciphertext = b''.join(ciphertext)

        elif mode == 'public':
            for i in range(int(math.ceil(len(byte_img)) / self.msglen)):
                ciphertext.append(self.public_cipher.encrypt(byte_img[int(i*self.msglen):int((i+1)*self.msglen)]))
            ciphertext = b''.join(ciphertext)
        
        cipher_image = np.frombuffer(ciphertext, dtype=img.dtype).reshape(h,w,d)

        return cipher_image

    # aes decryption using a key
    #INPUT
    # cipher_img        [h+1, w, 3]     encrypted image using aes
    #OUTPUT
    # decrypt_img       [h,w,3]         decrypted image using the key and aes mode    
    def decrypt(self,cipher_img,mode='private'):
        h,w,d = cipher_img.shape

        cipher_text = cipher_img.tobytes()
        decrypt_msg = []

        if mode == 'private':
            for i in range(int(math.ceil(len(cipher_text)) / self.msglen)):
                decrypt_msg.append(self.private_cipher.decrypt(cipher_text[int(i*self.msglen):int((i+1)*self.msglen)]))
            decrypt_msg = b''.join(decrypt_msg)
        elif mode == 'public':
            for i in range(int(math.ceil(len(cipher_text)) / self.msglen)):
                decrypt_msg.append(self.public_cipher.decrypt(cipher_text[int(i*self.msglen):int((i+1)*self.msglen)]))
            decrypt_msg = b''.join(decrypt_msg)
        
        decrypted_image = np.frombuffer(decrypt_msg, cipher_img.dtype).reshape(h,w,d)
        return decrypted_image

    def genkey(self,keysize):

        key = RSA.generate(keysize)
        with open('rsakey/privatekey.pem','wb') as f:
            f.write(key.export_key('PEM'))
        with open('rsakey/publickey.pem','wb') as f:
            f.write(key.publickey().export_key('PEM'))
        
# DES Encryption on images using Crypto Module
# ecb: electronic codebook. Most basic form
# cbc: cipher block chaining. Encryption of block is based on all previous blocks.
class DESimg():

    def __init__(self,key,mode='cbc'):
        self.mode = mode
        self.key = key
        self.iv = get_random_bytes(8) if mode == 'cbc' else get_random_bytes(0)
        if mode == 'cbc':
            self.cipher = DES.new(key,DES.MODE_CBC,self.iv)
        elif mode == 'ecb':
            self.cipher = DES.new(key,DES.MODE_ECB)
        else:
            print("ERROR with creating DES encryption")

    # given some data of any size, encrypt it using key
    def encrypt(self,data):
        bdata = data.tobytes()
        cdata = self.cipher.encrypt(bdata)
        cimg = np.frombuffer(cdata,dtype=data.dtype)
        return cimg

    # given some data of any size, decrypt it using key
    def decrypt(self,cdata):
        cipher = DES.new(self.key, DES.MODE_CBC) if self.mode == 'cbc' else DES.new(self.key,DES.MODE_ECB)
        pdata = cipher.decrypt(cdata.tobytes())
        return np.frombuffer(pdata, dtype=np.uint8)
        

# AES Encryption on images using Crypto module
# ecb: electronic codebook. Most basic form
# cbc: cipher block chaining. Encryption of block is based on all previous blocks.
class AESimg():

    def __init__(self,key,mode='ecb'):
        
        self.mode = mode
        self.key = key
        self.iv = get_random_bytes(32) if mode == 'cbc' else get_random_bytes(0)

        # key must be 32 bytes long
        if mode == 'cbc':
            self.cipher = AES.new(key,AES.MODE_CBC,self.iv)
        elif mode == 'ecb':
            self.cipher = AES.new(key,AES.MODE_ECB)

    # define key and key size
    #
    #INPUT:
    # img           [h,w,3]         numpy array as image
    #OUTPUT:
    # cimg          [h*w*3]       cipher image
    def encrypt(self, data):
        bdata = data.tobytes()
        cdata = self.cipher.encrypt(bdata)
        cimg = np.frombuffer(cdata,dtype=data.dtype)
        return cimg

    # aes decryption using a key
    #INPUT
    # cipher_img        [h*w*3]     encrypted image using aes
    #OUTPUT
    # decrypt_img       [h*w*3]         decrypted image using the key and aes mode
    def decrypt(self,cdata):
        cipher = AES.new(self.key, AES.MODE_CBC) if self.mode == 'cbc' else AES.new(self.key, AES.MODE_ECB)
        pdata = cipher.decrypt(cdata.tobytes())
        return np.frombuffer(pdata,dtype=np.uint8)

##########################################################################################

def getImageMatrix(imageName):
    im = Image.open(imageName) 
    pix = im.load()
    color = 1
    if type(pix[0,0]) == int:
      color = 0
    image_size = im.size 
    image_matrix = []
    for width in range(int(image_size[0])):
        row = []
        for height in range(int(image_size[1])):
                row.append((pix[width,height]))
        image_matrix.append(row)
    return image_matrix, image_size[0], image_size[1],color


def getImageMatrix_gray(imageName):
    im = Image.open(imageName).convert('LA')
    pix = im.load()
    image_size = im.size 
    image_matrix = []
    for width in range(int(image_size[0])):
        row = []
        for height in range(int(image_size[1])):
                row.append((pix[width,height]))
        image_matrix.append(row)
    return image_matrix, image_size[0], image_size[1]


# Arnold Cat Chaos Map based encryption
# requires optimization
def ArnoldCatTransform(img, num):
    # rows, cols, ch = img.shape
    # img_arnold = np.zeros([rows, cols, ch])

    rows, cols = img.shape
    img_arnold = np.zeros([rows, cols])
    
    for x in range(0, rows):
        for y in range(0, cols):
            # print(x, y)
            # print((x+y)%rows , (x+2*y)%cols)
            # print(img_arnold[x][y],img[(x+y)%rows][(x+2*y)%cols]   )
            img_arnold[x][y] = img[(x+y)%rows][(x+2*y)%cols]  
    return img_arnold    

# Arnold Cat Chaos Map based encryption
# requires optimization
def ArnoldCatReverseTransform(img, i):
    # rows, cols, ch = img.shape
    # img_arnold = np.zeros([rows, cols, ch])

    rows, cols = img.shape
    img_arnold = np.zeros([rows, cols])
    
    for x in range(0, rows):
        for y in range(0, cols):
            img_arnold[x][y] = img[(2*x-y)%rows][(y-x)%cols]  
    return img_arnold  

# encryption
def ArnoldCatEncryption(img, mask, key):
    for i in range (0,key):
        print(i)
        img = ArnoldCatTransform(img, i)
    return img

# decryption
def ArnoldCatDecryption(img, mask, key):
    # rows, cols = img.shape
    # dimension = rows
    # decrypt_it = dimension
    # if (dimension%2==0) and 5**int(round(log(dimension/2,5))) == int(dimension/2):
    #     decrypt_it = 3*dimension
    # elif 5**int(round(log(dimension,5))) == int(dimension):
    #     decrypt_it = 2*dimension
    # elif (dimension%6==0) and  5**int(round(log(dimension/6,5))) == int(dimension/6):
    #     decrypt_it = 2*dimension
    # else:
    #     decrypt_it = int(12*dimension/7)
    # print (key, dimension, decrypt_it)
    for i in range(0,key):
        print(i)
        img = ArnoldCatReverseTransform(img, i)

    return img

# Helper function for get Henon Map
def dec(bitSequence):
    decimal = 0
    for bit in bitSequence:
        decimal = decimal * 2 + int(bit)
    return decimal

# create Henon Map for Henon Encryption
def genHenonMap(dimensionX, dimensionY, key):
    x = key[0]
    y = key[1]
    sequenceSize = dimensionX * dimensionY * 8 #Total Number of bitSequence produced
    bitSequence = []    #Each bitSequence contains 8 bits
    byteArray = []      #Each byteArray contains m( i.e 512 in this case) bitSequence
    TImageMatrix = []   #Each TImageMatrix contains m*n byteArray( i.e 512 byteArray in this case)
    for i in range(sequenceSize):
        xN = y + 1 - 1.4 * x**2
        yN = 0.3 * x

        x = xN
        y = yN

        if xN <= 0.4:
            bit = 0
        else:
            bit = 1

        try:
            bitSequence.append(bit)
        except:
            bitSequence = [bit]

        if i % 8 == 7:
            decimal = dec(bitSequence)
            try:
                byteArray.append(decimal)
            except:
                byteArray = [decimal]
            bitSequence = []

        byteArraySize = dimensionY*8
        if i % byteArraySize == byteArraySize-1:
            try:
                TImageMatrix.append(byteArray)
            except:
                TImageMatrix = [byteArray]
            byteArray = []

    return TImageMatrix

# # Henon Encryption
# def HenonEncryption(imageMatrix,mask,key):
#     color = True
#     dimensionX, dimensionY, useless = imageMatrix.shape

#     # imageMatrix, dimension, color = getImageMatrix(imageName)
#     transformationMatrix = genHenonMap(dimensionX, dimensionY, key)
#     resultantMatrix = []
#     for i in range(dimensionX):
#         row = []
#         for j in range(dimensionY):
#             try:
#                 if color:
#                     row.append(tuple([int(transformationMatrix[i][j] ^ (255 if (x == 1) else int(x*255)))/255.0 for x in imageMatrix[i][j]]))
#                 else:
#                     row.append(transformationMatrix[i][j] ^ imageMatrix[i][j])
#             except:
#                 if color:
#                     row = [tuple([int(transformationMatrix[i][j] ^ (255 if (x == 1) else int(x*255)))/255.0 for x in imageMatrix[i][j]])]
#                 else :
#                     row = [int(transformationMatrix[i][j] ^ 255 if (x == 1) else int(x*256))/255.0 for x in imageMatrix[i][j]]
#         try:    
#             resultantMatrix.append(row)
#         except:
#             resultantMatrix = [row]
#     # if color:
#     #   im = Image.new("RGB", (dimensionX, dimensionY))
#     # else: 
#     #   im = Image.new("L", (dimensionX, dimensionY)) # L is for Black and white pixels

#     # pix = im.load()
#     # for x in range(dimensionX):
#     #     for y in range(dimensionY):
#     #         pix[x, y] = resultantMatrix[x][y]
    
#     return resultantMatrix



# # decrytion
# def HenonDecryption(imageMatrix,mask,key):
#     color = True
#     dimensionX, dimensionY, useless = imageMatrix.shape
#     # imageMatrix, dimension, color = getImageMatrix(imageName)
#     transformationMatrix = genHenonMap(dimensionX, dimensionY, key)
#     # pil_im = Image.open(imageNameEnc, 'r')
#     # imshow(np.asarray(pil_im))
#     henonDecryptedImage = []
#     for i in range(dimensionX):
#         row = []
#         for j in range(dimensionY):
#             try:
#                 if color:
#                     row.append(tuple([int(transformationMatrix[i][j] ^ (255 if (x == 1) else int(x*255)))/255.0 for x in imageMatrix[i][j]]))
#                 else:
#                     row.append(transformationMatrix[i][j] ^ imageMatrix[i][j])
#             except:
#                 if color:
#                     row = [tuple([int(transformationMatrix[i][j] ^ (255 if (x == 1) else int(x*255)))/255.0 for x in imageMatrix[i][j]])]
#                 else :
#                     row = [int(transformationMatrix[i][j] ^ 255 if (x == 1) else int(x*256))/256.0 for x in imageMatrix[i][j]]
#         try:
#             henonDecryptedImage.append(row)
#         except:
#             henonDecryptedImage = [row]
#     # if color:
#     #     im = Image.new("RGB", (dimensionX, dimensionY))
#     # else: 
#     #     im = Image.new("L", (dimensionX, dimensionY)) # L is for Black and white pixels

#     # pix = im.load()
#     # for x in range(dimensionX):
#     #     for y in range(dimensionY):
#     #         pix[x, y] = henonDecryptedImage[x][y]
#     # im.save(imageNameEnc.split('_')[0] + "_HenonDec.png", "PNG")
#     return henonDecryptedImage


#incase we need to work with 0-255 int range colors

# Henon Encryption
def HenonEncryption(imageMatrix,mask,key):
    color = True
    dimensionX, dimensionY, useless = imageMatrix.shape

    transformationMatrix = genHenonMap(dimensionX, dimensionY, key)
    resultantMatrix = []
    for i in range(dimensionX):
        row = []
        for j in range(dimensionY):
            try:
                if color:
                    row.append(tuple([transformationMatrix[i][j] ^ int(x) for x in imageMatrix[i][j]]))
                else:
                    row.append(transformationMatrix[i][j] ^ imageMatrix[i][j])
            except:
                if color:
                    row = [tuple([transformationMatrix[i][j] ^ int(x) for x in imageMatrix[i][j]])]
                else :
                    row = [transformationMatrix[i][j] ^ int(x) for x in imageMatrix[i][j]]
        try:    
            resultantMatrix.append(row)
        except:
            resultantMatrix = [row]
    
    
    return resultantMatrix



# decrytion
def HenonDecryption(imageMatrix,mask,key):
    color = True
    dimensionX, dimensionY, useless = imageMatrix.shape
    
    transformationMatrix = genHenonMap(dimensionX, dimensionY, key) 
    henonDecryptedImage = []
    for i in range(dimensionX):
        row = []
        for j in range(dimensionY):
            try:
                if color:
                    row.append(tuple([transformationMatrix[i][j] ^ int(x) for x in imageMatrix[i][j]]))
                else:
                    row.append(transformationMatrix[i][j] ^ imageMatrix[i][j])
            except:
                if color:
                    print(i, j, imageMatrix[i][j])
                    row = [tuple([transformationMatrix[i][j] ^ int(x) for x in imageMatrix[i][j]])]
                else :
                    row = [transformationMatrix[i][j] ^ int(x) for x in imageMatrix[i][j]]
        try:
            henonDecryptedImage.append(row)
        except:
            henonDecryptedImage = [row]
   
    return henonDecryptedImage