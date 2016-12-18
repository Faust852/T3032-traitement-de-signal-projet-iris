# import the necessary packages
import numpy as np
import cv2
import cv
import Image
from itertools import izip
from matplotlib import pyplot as plt
from math import hypot

def locateIris (path):
    base = cv2.imread(path)
    origin = base.copy()

    gray_scale = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    iris = base.copy()
    pupil = base.copy()

    clean_gray = cv2.GaussianBlur(gray_scale, (3, 3), 0)
    clean_pupil = clean_gray.copy()
    clean_iris = clean_gray.copy()

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

    circles_pupil = cv2.HoughCircles(edges_pupil, cv2.cv.CV_HOUGH_GRADIENT, 2, 400, \
                                     param1=20, param2=30, minRadius=40, maxRadius=100)

    for circle in circles_pupil[0, :]:
        cv2.circle(origin, (circle[0], circle[1]), circle[2], (0, 0, 0), -1)
        cv2.circle(origin, (circle[0], circle[1]), circle[2], (0, 0, 0), 2)
        cv2.circle(origin, (circle[0], circle[1]), 2, (0, 0, 0), 2)
        # print int(circle[0])
        # print int(circle[1])
        # print int(circle[2])

        circle[2] = circle[2] + 60.0

        cv2.circle(origin, (circle[0], circle[1]), circle[2], (0, 0, 0), 2)

        x1 = int(circle[0] - circle[2] + 2)
        y1 = int(circle[1] - circle[2] + 2)
        x2 = int(circle[0] + circle[2] - 2)
        y2 = int(circle[1] + circle[2] - 2)

    #circles_iris = cv2.HoughCircles(edges_iris, cv2.cv.CV_HOUGH_GRADIENT, 2, 400, \
    #                               param1=20, param2=30, minRadius=90, maxRadius=200)

    x, y, r = circles_pupil[0, :][0]
    rows, cols, channel = origin.shape
    for i in range(cols):
        for j in range(rows):
            if hypot(i - x, j - y) > r:
                origin[j, i] = 0
    #
    # for circle in circles_iris[0, :]:
    #     cv2.circle(origin, (circle[0], circle[1]), circle[2], (0, 0, 0), 2)
    #     cv2.circle(origin, (circle[0], circle[1]), 2, (40, 40, 40), 2)
    #     # print int(circle[0])
    #     # print int(circle[1])
    #     # print int(circle[2])
    x1 = int(circle[0] - 100)#circle[2]-5)
    y1 = int(circle[1] - 100)#circle[2]-5)
    x2 = int(circle[0] + 100)#circle[2]+5)
    y2 = int(circle[1] + 100)#circle[2]+5)

    crop_origin = cropIris(origin, x1, y1, x2, y2)

    ########WHYYYY


    return {'im':crop_origin,'rad_iris':circle[2],"x":circle[0],"y":circle[1]}

def irisProcessing(image, kernel) :
    processedIris = image
    processedIris = cv2.cvtColor(processedIris, cv2.COLOR_BGR2GRAY)
    cl1 = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    clahe = cl1.apply(processedIris)
    ret, thresh1 = cv2.threshold(clahe, 170, 255, cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)

    return opening


def findPatern(image) :
    keys = image
    fast = cv2.FastFeatureDetector()
    kp = fast.detect(keys, None)
    img2 = cv2.drawKeypoints(keys, kp, color=(255, 0, 0))
    fast.setBool('nonmaxSuppression', 0)
    kp = fast.detect(keys, None)

    img3 = cv2.drawKeypoints(keys, kp, color=(255, 0, 0))

    return img2

def normalize(image, rad):
    imgSize = cv2.cv.GetSize(image)
    c = (float(imgSize[0]/2.0), float(imgSize[1]/2.0))
    imgRes = cv2.cv.CreateImage((rad*3, int(360)), 8, 3)
    cv2.cv.LogPolar(image,imgRes,c,60.0, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS)
    return (imgRes)


def cropIris (image, x1, y1, x2, y2):
    crop = image[y1:y2, x1:x2]
    if crop is not None:
        return crop
    return None

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


def mse (imageA, imageB):

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    if err != 0 :
        return err
    else :
        return 100000
