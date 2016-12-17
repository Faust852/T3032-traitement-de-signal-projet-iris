# import the necessary packages
import numpy as np
import cv2
import Image
from itertools import izip

def locateIris (image, kernel):
    #converti en rgb -> gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #floutte
    cl1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe = cl1.apply(gray)
    #floutte encore plus
    medianblur = cv2.medianBlur(clahe, 19)
    #blur = cv2.medianBlur(medianblur, 19)
    erosion = cv2.morphologyEx(medianblur, cv2.MORPH_OPEN, kernel)
    #trouve les edges entre les variations de couleurs
    edges = cv2.Canny(erosion, 30, 60)
    #trouve l'iris (les cercles)
    circles = cv2.HoughCircles(edges, cv2.cv.CV_HOUGH_GRADIENT, 1, 3000,
                               param1=30, param2=15, minRadius=30, maxRadius=150)
    #dessine un cercle autour de ceux trouvés
    for circle in circles[0, :]:
        cv2.circle(image, (circle[0], circle[1]), circle[2], (255, 255, 255), 1)
        cv2.circle(image, (circle[0], circle[1]), 2, (255, 255, 255), 2)
        # print int(circle[0])
        # print int(circle[1])
        # print int(circle[2])
        x1 = int(circle[0] - 100)#circle[2])
        y1 = int(circle[1] - 100)#circle[2])
        x2 = int(circle[0] + 100)#circle[2])
        y2 = int(circle[1] + 100)#circle[2])

    return {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}

#decoupe l'image autour du cercle
def cropIris (image, x1, y1, x2, y2):
    crop = image[y1:y2, x1:x2]
    if crop is not None:
        return crop
    return None

#compare les images
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
