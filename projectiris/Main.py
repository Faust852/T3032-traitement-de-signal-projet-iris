import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import utils
import numpy as np

## Read Image
image2 = cv2.imread('S1001R02.jpg')
image = cv2.imread('iris3.jpg')

imageBackup = image.copy()

## Convert to 1 channel only grayscale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
## CLAHE Equalization



cl1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe = cl1.apply(gray)
## medianBlur the image to remove noise
medianblur = cv2.medianBlur(clahe, 19)
blur = cv2.medianBlur(medianblur, 19)



edges = cv2.Canny(blur, 30, 60)

## Detect Circlesr
circles = cv2.HoughCircles(edges , cv2.cv.CV_HOUGH_GRADIENT,1,3000,
                            param1=30,param2=15,minRadius=50,maxRadius=200)

for circle in circles[0,:]:
    # draw the outer circle
    cv2.circle(image,(circle[0],circle[1]),circle[2],(0,255,0),2)

    # draw the center of the circle
    cv2.circle(image,(circle[0],circle[1]),2,(0,0,255),3)
    print int(circle[0])
    print int(circle[1])
    print int(circle[2])
    x1 = int(circle[0] - circle[2])
    y1 = int(circle[1] - circle[2])
    x2 = int(circle[0] + circle[2])
    y2 = int(circle[1] + circle[2])

    crop = imageBackup[y1:y2,x1:x2]
    #if crop is not None:
        #cv2.imshow('test', crop)

# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib

image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

# show our image
plt.figure()
plt.axis("off")
plt.imshow(crop)

# reshape the image to be a list of pixels
image = image.reshape((image.shape[0] * image.shape[1], 3))

# cluster the pixel intensities
clt = KMeans(10)
clt.fit(image)

# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = utils.centroid_histogram(clt)
bar = utils.plot_colors(hist, clt.cluster_centers_)

# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()