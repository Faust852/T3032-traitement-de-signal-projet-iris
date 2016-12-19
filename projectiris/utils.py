#From Micciche David - 2016
#For EPHEC - Signal Processing Assignement

import numpy as np
import cv2
import Image
from itertools import izip
from matplotlib import pyplot as plt
from math import hypot
from math import ceil, exp, pi
from math import log, cos, sin
from scipy import ndimage

kernel = np.ones((5,5),np.uint8)

def wrapperIris (path):
    im1 = cv2.imread(path)
    cv2.imshow('origin1', im1)


    im1 = locateIris(path)
    im1_smg = irisProcessing(im1['im'], kernel)
    cv2.imshow('segm1', im1_smg)


    im1_norm = normalize(im1_smg, im1['rad_iris'])
    cv2.imshow('norm1', im1_norm)


    _,im1_bina = cv2.threshold(im1_norm, 170, 255, cv2.THRESH_BINARY)
    cv2.imshow('bin1', im1_bina)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return im1, im1_smg, im1_norm, im1_bina

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

    # edges_iris = cl1.apply(clean_iris)
    # edges_iris = cv2.GaussianBlur(edges_iris, (15, 15), 0)
    # edges_iris = cv2.GaussianBlur(edges_iris, (15, 15), 0)
    # edges_iris = cv2.Canny(edges_iris, 30, 60)
    # edges_iris = cv2.GaussianBlur(edges_iris, (15, 15), 0)

    #HoughCircles is a function that is looking for circle shape in the image
    #We use a small radius because we only need to isolate the pupil
    circles_pupil = cv2.HoughCircles(edges_pupil, cv2.HOUGH_GRADIENT, 2, 400, \
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
    cl1 = cv2.createCLAHE(clipLimit=22.0, tileGridSize=(8, 8))
    clahe = cl1.apply(processedIris)
    #ret, thresh1 = cv2.threshold(clahe, 170, 255, cv2.THRESH_BINARY)
    #opening = cv2.morphologyEx(clahe, cv2.MORPH_OPEN, kernel)
    return clahe

##  TO DO
#input : image (nparray)
#output : image (nparray)
# The function is used to locate pattern on a picture
##
def findPattern(image) :
    orb = cv2.ORB_create()
    kp = orb.detect(image, None)
    kp, des = orb.compute(image, kp)
    image = cv2.drawKeypoints(image, kp,image, color=(0, 255, 0), flags=0)
    return image

##
#TEMP FONCTION
#
##
def comparePattern (image1, image2) :
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    print len(matches)
    img3 = cv2.drawMatches(image1, kp1, image2, kp2, matches[:1000],None, flags=2)
    plt.imshow(img3), plt.show()
    return len(matches)

def binaryComparison(img1, img2):
    labelarray1, precount1 = ndimage.measurements.label(img1)
    labelarray2, precount2 = ndimage.measurements.label(img2)
    if precount1 > precount2:
        numberOfPixels = precount1
    else:
        numberOfPixels = precount2
    results = cv2.bitwise_and(img1, img2, None, None)
    cv2.imshow('result', results)
    labelarray, particle_count = ndimage.measurements.label(results)
    final = particle_count
    percentage = ((float(final)) / numberOfPixels)*100
    return percentage

def comparePatternSift (image1, image2) :
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good)>10:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w, d = image1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        image2 = cv2.polylines(image2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print "Not enough matches are found - %d/%d" % (len(good), 10)
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
    singlePointColor = None,
    matchesMask = matchesMask,  # draw only inliers
    flags = 2)
    img3 = cv2.drawMatches(image1, kp1, image2, kp2, good, None, **draw_params)
    plt.imshow(img3, 'gray'), plt.show()



# input : nparray & kernel (see irisProcessing
# output : nparray
# contour the image, easier to extract
##
def drawContour(image, kernel):
    ret, image = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.erode(image, kernel, iterations=1)
    return image
##
# input : nparray and int
# output : nparray
# Return a normalized image of the iris (a rectangle) Much easier to compare and match
##
def normalize(image, rad):
    tmp = np.zeros((3*rad, image.shape[0], 3),np.uint8)

    c = (float(image.shape[0] / 2.0), float(image.shape[1] / 2.0))
    image = cv2.logPolar(image,(image.shape[0] / 2, image.shape[1] / 2), 42, cv2.WARP_FILL_OUTLIERS)
    #imgRes = logpolar_naive(image, float(imgSize[0]/2.0), float(imgSize[1]/2.0))
    #imgRes = ndimage.rotate(imgRes, 90)
    # mask = cv2.imread('mask.jpg')
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # imgRes = cv2.bitwise_and(imgRes, imgRes, mask=mask)
    return (image)

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

def logpolar_naive (image, i_0, j_0, p_n=None, t_n=None):
    '''"Naive" implementation of the log-polar transform in Python.
        Arguments:
        image
            The input image.
        i_0, j0
            The center of the transform.
        p_n, t_n
            Optional. Dimensions of the output transform. If any are None,
            suitable defaults are used.
        Returns:
        The log-polar transform for the input image.
    '''
    # Shape of the input image.
    (i_n, j_n) = image.shape[:2]
    # The distance d_c from the transform's focus (i_0, j_0) to the image's
    # farthest corner (i_c, j_c). This is used below as the default value for
    # p_n, and also to calculate the iteration step across the transform's p
    # dimension.
    i_c = max(i_0, i_n - i_0)
    j_c = max(j_0, j_n - j_0)
    d_c = (i_c ** 2 + j_c ** 2) ** 0.5
    if p_n == None:
        # The default value to p_n is defined as the distance d_c.
        p_n = int(ceil(d_c))
    if t_n == None:
        # The default value to t_n is defined as the width of the image.
        t_n = j_n
    # The scale factors determine the size of each "step" along the transform.
    p_s = log(d_c) / p_n
    t_s = 2.0 * pi / t_n
    # The transform's pixels have the same type and depth as the input's.
    transformed = np.zeros((p_n, t_n) + image.shape[2:], dtype=image.dtype)
    # Scans the transform across its coordinate axes. At each step calculates
    # the reverse transform back into the cartesian coordinate system, and if
    # the coordinates fall within the boundaries of the input image, takes that
    # cell's value into the transform.
    for p in range(0, p_n):
        p_exp = exp(p * p_s)
        for t in range(0, t_n):
            t_rad = t * t_s

            i = int(i_0 + p_exp * sin(t_rad))
            j = int(j_0 + p_exp * cos(t_rad))

            if 0 <= i < i_n and 0 <= j < j_n:
                transformed[p, t] = image[i, j]

    return transformed
