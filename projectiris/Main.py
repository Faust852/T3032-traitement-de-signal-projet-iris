
import numpy as np
import cv2
import utils

i1,s1,n1,b1= utils.wrapperIris('./images/NIR_2/002_1_1.bmp') #OEIL 1
i2,s2,n2,b2 = utils.wrapperIris('./images/NIR_2/003_2_3.bmp') #OEIL 2

v1 = utils.comparePattern(s1, s2)
v2 = utils.comparePattern(n1, n2)
v3 = utils.comparePattern(b1, b2)
v4 = utils.binaryComparison(b1,b2)
print v4
if ((v1 > 150) & (v2 > 30) & (v3 > 10)) :
    print "MATCHING"
else:
    print "NOT MATCHING"
