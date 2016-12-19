
import numpy as np
import cv2
import utils


i1,s1,n1,b1= utils.wrapperIris('./images/NIR_2/003_2_1.bmp')
i2,s2,n2,b2 = utils.wrapperIris('./images/NIR_2/003_1_2.bmp')

v0 = utils.comparePattern(i1['im'], i2['im'])
v1 = utils.comparePattern(s1, s2)
v2 = utils.comparePattern(n1, n2)
v3 = utils.comparePattern(b1, b2)
v4 = utils.binaryComparison(b1,b2)
print v4
if ((v1 > 150) & (v2 > 50) & (v3 > 20)) :
    print "MATCHING"
else:
    print "NOT MATCHING"
