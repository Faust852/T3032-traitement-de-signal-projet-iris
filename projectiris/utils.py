import numpy as np
import cv2
import Image
import utils
from matplotlib import pyplot as plt
from math import hypot
from itertools import izip
from scipy import ndimage
from sklearn.preprocessing import normalize

kernel = np.ones((3,3),np.uint8)

path = './images/NIR_2/001_1_2.bmp'
path2 = './images/NIR_2/001_2_3.bmp'

img = utils.locateIris(path)
img2 = utils.locateIris(path2)

processed_img = utils.irisProcessing(img['im'], kernel)
processed_img2 = utils.irisProcessing(img2['im'], kernel)

cv2.imwrite("tmp1.jpg", img["im"])
temp1 = cv2.imread("tmp1.jpg")

cv2.imwrite("tmp2.jpg", img2["im"])
temp2 = cv2.imread("tmp2.jpg")

test = utils.normalize(temp1,int(img["rad_iris"]))

test2 = utils.normalize(temp2,int(img2["rad_iris"]))

var = np.asarray(test[:,:] )

var2 = np.asarray(test2[:,:] )

cv2.imwrite("tmpthresh.jpg", var)
cv2.imwrite("tmpthresh2.jpg", var2)

im = cv2.imread("tmpthresh.jpg")
im2 = cv2.imread("tmpthresh2.jpg")

im = ndimage.rotate(im, 90)
im2 = ndimage.rotate(im2, 90)

im = utils.irisProcessing(im, kernel)
im2 = utils.irisProcessing(im2, kernel)
cv2.imshow("im", im)
cv2.imshow("im2", im2)


# pattern = utils.findPattern(im_clahe)
# pattern2 = utils.findPattern(im_clahe2)


utils.comparePattern(im, im2)




cv2.waitKey(0)

# image_threshold = .5
#
# s = ndimage.generate_binary_structure(2,2)
#
# labeled_array, num_features = ndimage.label(processed_img, structure=s)
# print(num_features)
#
# sizes = ndimage.sum(processed_img,labeled_array,range(1,num_features+1))
# map = np.where(sizes==sizes.max())[0] + 1
# mip = np.where(sizes==sizes.min())[0] + 1
#
# max_index = np.zeros(num_features + 1, np.uint8)
# max_index[map] = 1
# max_feature = max_index[labeled_array]
#
# min_index = np.zeros(num_features + 1, np.uint8)
# min_index[mip] = 1
# min_feature = min_index[labeled_array]

# plt.subplot(121)
# plt.imshow(thresh1)
# plt.subplot(122)
# plt.imshow(thresh2)
# plt.show()

cv2.waitKey(0)
