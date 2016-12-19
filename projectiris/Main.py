
import numpy
import cv2
import utils
import Image
import glob
import argparse
from scipy import ndimage
from matplotlib import pyplot as plt

i1,s1,n1,b1= utils.wrapperIris('./images/NIR_2/002_1_3.bmp')
i2,s2,n2,b2 = utils.wrapperIris('./images/NIR_2/001_1_2.bmp')


v1 = utils.comparePattern(s1, s2)
utils.comparePattern(n1, n2)
if (v1 > 199) :
    print "MATCHING"
else:
    print "NOT MATCHING"

cv2.waitKey(0)
