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

kernel = np.ones((2,2),np.uint8)

path = './images/NIR_2/001_2_2.bmp'
path2 = './images/NIR_2/003_2_4.bmp'

img = utils.locateIris(path)
img2 = utils.locateIris(path2)

cv2.imshow("img", img['im'])
cv2.imshow("img2", img2['im'])

processed_img = utils.irisProcessing(img['im'], kernel)
processed_img2 = utils.irisProcessing(img2['im'], kernel)

test = utils.normalize(cv.fromarray(img['im']),int(img['rad_iris']))
cv2.imshow("test", np.asarray(test[:,:] ))

test2 = utils.normalize(cv.fromarray(img2['im']),int(img2['rad_iris']))
cv2.imshow("test2", np.asarray(test2[:,:] ))

pattern = utils.findPatern(processed_img)
pattern2 = utils.findPatern(processed_img2)

cv2.waitKey(0)

image_threshold = .5

s = ndimage.generate_binary_structure(2,2)

labeled_array, num_features = ndimage.label(processed_img, structure=s)
print(num_features)

sizes = ndimage.sum(processed_img2,labeled_array,range(1,num_features+1))
map = np.where(sizes==sizes.max())[0] + 1
mip = np.where(sizes==sizes.min())[0] + 1

max_index = np.zeros(num_features + 1, np.uint8)
max_index[map] = 1
max_feature = max_index[labeled_array]

min_index = np.zeros(num_features + 1, np.uint8)
min_index[mip] = 1
min_feature = min_index[labeled_array]

plt.subplot(121)
plt.imshow(pattern)
plt.subplot(122)
plt.imshow(pattern2)
plt.show()

cv2.waitKey(0)
