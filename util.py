
import numpy as np



# encrypt image using encryption mode. if loc or size is None, encrypt entire image
# img       [h,w,3]         numpy array of image
# key       scalar          string used for encryption key
def encrypt_image(img,cipher,loc=(0,0),size=128):

    subimg = img[loc[0]:loc[0] + size, loc[1]:loc[1] + size]

    cipimg = cipher.encrypt(subimg)

    newimg = np.copy(img)
    if subimg.shape == cipimg.shape:
        newimg[loc[0]:loc[0]+size, loc[1]:loc[1]+size] = cipimg
    else:
        newimg[loc[0]:loc[0]+size+1, loc[1]:loc[1]+size] = cipimg

    return newimg
