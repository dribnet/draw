import numpy as np
from skimage import data
from skimage.color import rgb2lab, lab2lch, lch2lab, lab2rgb

def rgb2lch(im):
    # convert image to lch
    img_lab = rgb2lab(im)
    img_lch = lab2lch(img_lab)
    return img_lch

def lch2rgb(im):
    # convert image to lch
    img_lab = lch2lab(im)
    img_rgb = lab2rgb(img_lab)
    return img_rgb

def scale2lab(im255):
    # convert image to lch
    img_sca = im255 * (1.0 / 255.0)
    return rgb2lab(img_sca)

def rgb2scaled_lab(im):
    img_lab = rgb2lab(im)
    img_lab[:,:,0] = img_lab[:,:,0] / 100.0
    img_lab[:,:,1] = (img_lab[:,:,1] + 100) / 200.0
    img_lab[:,:,2] = (img_lab[:,:,2] + 100) / 200.0
    return img_lab.astype(np.float32)

def scaled_lab2rgb(img):
    img_lab = np.copy(img).astype(np.float64)
    img_lab[:,:,0] = img_lab[:,:,0] * 100.0
    img_lab[:,:,1] = (img_lab[:,:,1] * 200.0) - 100.0
    img_lab[:,:,2] = (img_lab[:,:,2] * 200.0) - 100.0
    im = lab2rgb(img_lab)
    return im.astype(np.float32)

def scaled255_lab2rgb(img):
    im = 255.0 * scaled_lab2rgb((1.0 / 255.0) * img)
    return im.astype(np.uint8)
