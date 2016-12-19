
import numpy as np
import cv2
import Image
import utils
from matplotlib import pyplot as plt
from math import hypot
from itertools import izip
from scipy import ndimage
from sklearn.preprocessing import normalize

kernel = np.ones((5,5),np.uint8)

im1 = cv2.imread('./images/NIR_2/001_1_1.bmp')
im2 = cv2.imread('./images/NIR_2/001_2_2.bmp')

im1 = utils.locateIris('./images/NIR_2/001_1_1.bmp')
im2 = utils.locateIris('./images/NIR_2/001_1_2.bmp')
im1_smg = utils.irisProcessing(im1['im'], kernel)
im2_smg = utils.irisProcessing(im2['im'], kernel)

im1_norm = utils.normalize(im1_smg, im1['rad_iris'])
im2_norm = utils.normalize(im2_smg, im2['rad_iris'])

_,im1_bina = cv2.threshold(im1_norm, 190, 255, cv2.THRESH_BINARY)
_,im2_bina = cv2.threshold(im2_norm, 190, 255, cv2.THRESH_BINARY)

cv2.imshow('ima',im1['im'])
cv2.imshow('imb',im2['im'])
cv2.imshow('im',im1_norm)
cv2.imshow('im2',im2_norm)


cv2.waitKey(0)
cv2.destroyAllWindows()
