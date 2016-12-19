
import numpy
import cv2
import utils
import Image
import glob
import argparse
from scipy import ndimage
from matplotlib import pyplot as plt

i1,s1,n1,b1= utils.wrapperIris('./images/NIR_2/003_1_3.bmp')
i2,s2,n2,b2 = utils.wrapperIris('./images/NIR_2/003_1_2.bmp')

#
# v1 = utils.comparePattern(s1, s2)
# if (v1 > 199) :
#     print "MATCHING"
# else:
#     print "NOT MATCHING"

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
label_array1, n_features1 =  ndimage.label(b1)
print n_features1
label_array2, n_features2 =  ndimage.label(b2)
print n_features2
plt.subplot(121)
plt.imshow(label_array2)
plt.subplot(122)
plt.imshow(label_array1)
plt.show()

cv2.waitKey(0)
