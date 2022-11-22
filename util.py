
import numpy as np



# encrypt parts of the image given by img_mask
# due to block size, the img_mask will be adjusted if it is not some multiple of the encryption
# block size.
def encrypt_image_mask(img, img_mask):



    return 

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


# fix mask
def fix_mask(mask, blocksize):

    n = np.lcm(blocksize, 24)
    if np.sum(mask) % n != 0:
        k = np.sum(mask) % n
        ones = mask[mask]

        ones[-k:] = False
        mask[mask] = ones

    return mask
