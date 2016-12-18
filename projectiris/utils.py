#From Micciche David - 2016
#For EPHEC - Signal Processing Assignement

import numpy as np
import cv2
import cv
import Image
from itertools import izip
from matplotlib import pyplot as plt
from math import hypot

##
#input : the path of an image (string)
#output : processed image (nparray)
#
#The fonction processes eye. It localized the pupil, and the outer circle or the iris
#It then blacks out the useless information and crops the iris
#It returns the processed iris without uneeded information
##
def locateIris (path):
    base = cv2.imread(path)
    origin = base.copy()
    #convert in grayscale because numpy doesn't like color
    gray_scale = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    iris = base.copy()
    pupil = base.copy()
    #1st slight blur to remove unwanted artifact
    clean_gray = cv2.GaussianBlur(gray_scale, (3, 3), 0)
    clean_pupil = clean_gray.copy()
    clean_iris = clean_gray.copy()

    #CLAHE is used to strengthen the contrast, so we can see the outer border of the iris better
    #Gaussian Blur is used to remove unwanted artifact
    #Canny is a thresholding tool allowing us to better see circles in the image
    cl1 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    edges_pupil = cl1.apply(clean_pupil)
    edges_pupil = cv2.GaussianBlur(edges_pupil, (15, 15), 0)
    edges_pupil = cv2.Canny(edges_pupil, 30, 60)
    edges_pupil = cv2.GaussianBlur(edges_pupil, (15, 15), 0)

    edges_iris = cl1.apply(clean_iris)
    edges_iris = cv2.GaussianBlur(edges_iris, (15, 15), 0)
    edges_iris = cv2.GaussianBlur(edges_iris, (15, 15), 0)
    edges_iris = cv2.Canny(edges_iris, 30, 60)
    edges_iris = cv2.GaussianBlur(edges_iris, (15, 15), 0)

    #HoughCircles is a function that is looking for circle shape in the image
    #We use a small radius because we only need to isolate the pupil
    circles_pupil = cv2.HoughCircles(edges_pupil, cv2.cv.CV_HOUGH_GRADIENT, 2, 400, \
                                     param1=20, param2=30, minRadius=40, maxRadius=100)

    #Black out the inner pupil (remove lights, and other artifacts)
    for circle in circles_pupil[0, :]:
        cv2.circle(origin, (circle[0], circle[1]), circle[2], (0, 0, 0), -1)
        cv2.circle(origin, (circle[0], circle[1]), circle[2], (0, 0, 0), 2)
        cv2.circle(origin, (circle[0], circle[1]), 2, (0, 0, 0), 2)

        circle[2] = circle[2] + 60.0

        cv2.circle(origin, (circle[0], circle[1]), circle[2], (0, 0, 0), 2)

        # x1 = int(circle[0] - circle[2] + 2)
        # y1 = int(circle[1] - circle[2] + 2)
        # x2 = int(circle[0] + circle[2] - 2)
        # y2 = int(circle[1] + circle[2] - 2)

    #We don't use the function for the iris anymore, it's easier to use a fixed value so we have perfect circles everytime
    #circles_iris = cv2.HoughCircles(edges_iris, cv2.cv.CV_HOUGH_GRADIENT, 2, 400, \
    #                               param1=20, param2=30, minRadius=90, maxRadius=200)

    # blacken the outer part of the iris
    x, y, r = circles_pupil[0, :][0]
    rows, cols, channel = origin.shape
    for i in range(cols):
        for j in range(rows):
            if hypot(i - x, j - y) > r:
                origin[j, i] = 0

    # for circle in circles_iris[0, :]:
    #     cv2.circle(origin, (circle[0], circle[1]), circle[2], (0, 0, 0), 2)
    #     cv2.circle(origin, (circle[0], circle[1]), 2, (40, 40, 40), 2)

    # hardcoded value,
    x1 = int(circle[0] - 100)#circle[2]-5)
    y1 = int(circle[1] - 100)#circle[2]-5)
    x2 = int(circle[0] + 100)#circle[2]+5)
    y2 = int(circle[1] + 100)#circle[2]+5)

    #crop so we don't have extra black around the picture
    crop_origin = cropIris(origin, x1, y1, x2, y2)
    return {'im':crop_origin,'rad_iris':circle[2],"x":circle[0],"y":circle[1]}

##
#input : image (nparray) and kernel (np.ones((3,3),np.uint8))
#output : processed image (nparray)
# The function is used to help pattern finding
##
def irisProcessing(image, kernel) :
    processedIris = image
    processedIris = cv2.cvtColor(processedIris, cv2.COLOR_BGR2GRAY)
    cl1 = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    clahe = cl1.apply(processedIris)
    ret, thresh1 = cv2.threshold(clahe, 170, 255, cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    return opening

##  TO DO
#input : image (nparray)
#output : image (nparray)
# The function is used to locate pattern on a picture
##
def findPatern(image) :
    keys = image
    fast = cv2.FastFeatureDetector()
    kp = fast.detect(keys, None)
    img2 = cv2.drawKeypoints(keys, kp, color=(255, 0, 0))
    return img2

##
#TEMP FONCTION
#
##
def comparePatternORB (image1, image2) :
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], flags=2)
    plt.imshow(img3), plt.show()


##
# input : nparray & kernel (see irisProcessing
# output : nparray
# contour the image, easier to extract
##
def drawContour(image, kernel):
    im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(im_gray,1,255,0)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0,255,0), 1)
    return image

##
# input : nparray and int
# output : nparray
# Return a normalized image of the iris (a rectangle) Much easier to compare and match
##
def normalize(image, rad):
    imgSize = cv2.cv.GetSize(image)
    c = (float(imgSize[0]/2.0), float(imgSize[1]/2.0))
    imgRes = cv2.cv.CreateImage((rad*3, int(360)), 8, 3)
    cv2.cv.LogPolar(image,imgRes,c,60.0, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS)
    return (imgRes)

##
#input : nparray, int, int, int, int
#output : nparray
# Used to crop the iris
##
def cropIris (image, x1, y1, x2, y2):
    crop = image[y1:y2, x1:x2]
    if crop is not None:
        return crop
    return None

##
# TO USE IF THE REST FAIL :(:(:(:(:(
#
#
##
def compareImages(i1, i2) :
    assert i1.mode == i2.mode, "Different kinds of images."
    assert i1.size == i2.size, "Different sizes."

    pairs = izip(i1.getdata(), i2.getdata())
    if len(i1.getbands()) == 1:
        # for gray-scale jpegs
        dif = sum(abs(p1 - p2) for p1, p2 in pairs)
    else:
        dif = sum(abs(c1 - c2) for p1, p2 in pairs for c1, c2 in zip(p1, p2))

    ncomponents = i1.size[0] * i1.size[1] * 3
    res = (dif / 255.0 * 100) / ncomponents
    if res != 0 :
        return res
    else :
        return 1000


##
# TO USE IF THE REST FAIL :(:(:(:(:(
#
#
##
def mse (imageA, imageB):

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    if err != 0 :
        return err
    else :
        return 100000
