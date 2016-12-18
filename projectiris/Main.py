import numpy as np
import cv2
import cv
import Image
import utils
from matplotlib import pyplot as plt
from math import hypot
from itertools import izip
from scipy import ndimage
from sklearn.preprocessing import normalize

kernel = np.ones((3,3),np.uint8)

path = './images/NIR_2/001_2_2.bmp'
path2 = './images/NIR_2/001_2_4.bmp'

img = utils.locateIris(path)
img2 = utils.locateIris(path2)

processed_img = utils.irisProcessing(img['im'], kernel)
processed_img2 = utils.irisProcessing(img2['im'], kernel)

cv2.imwrite("tmp1.jpg", img["im"])
temp1 = cv2.imread("tmp1.jpg")
cv2.imshow("tmp", temp1)

# cv2.imwrite("tmp2.jpg", img2["im"])
# temp2 = cv2.imread("tmp2.jpg")
# cv2.imshow("tmp2", temp2)

test = utils.normalize(cv.fromarray(temp1),int(img["rad_iris"]))

var = np.asarray(test[:,:] )
cv2.imshow("test", var)


cv2.imwrite("tmpthresh.jpg", var)
tmpthresh = cv2.imread("tmpthresh.jpg")
gray = cv2.cvtColor(tmpthresh,cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cv2.imshow("tsk",opening)

# test2 = utils.normalize(cv.fromarray(temp2),int(img2['rad_iris']))
# cv2.imshow("test2", np.asarray(test2[:,:] ))

pattern = utils.findPatern(processed_img)
pattern2 = utils.findPatern(processed_img2)

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
# plt.imshow(pattern)
# plt.subplot(122)
# plt.imshow(pattern2)
# plt.show()

cv2.waitKey(0)
