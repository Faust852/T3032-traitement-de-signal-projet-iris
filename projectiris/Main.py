import cv2
import utils
import numpy
import glob
import Image
from itertools import izip

imageFolderPath = '/home/faust/PycharmProjects/T3032-traitement-de-signal-projet-iris'
imagePath = glob.glob('*.JPG')
im_array = numpy.array( [numpy.array(Image.open(imagePath[i]).convert('L'), 'f') for i in range(len(imagePath))] )

kernel = numpy.ones((5,5),numpy.uint8)

origin = cv2.imread('img1.JPG')

valueComp = []
valueMSE = []

res = []
i = 0
for picture in glob.glob('*.JPG') :
    im_cv = cv2.imread(picture)
    imOrigin = im_cv.copy()

    crop_array = utils.locateIris(im_cv, kernel)

    res.append(utils.cropIris(imOrigin, crop_array['x1'],crop_array['y1'],crop_array['x2'],crop_array['y2']))

    i1_np = utils.cropIris(origin, crop_array['x1'],crop_array['y1'],crop_array['x2'],crop_array['y2'])
    i1 = Image.fromarray(i1_np, 'RGB')

    valueComp.append(utils.compareImages(i1, Image.fromarray(res[i], 'RGB')))
    valueMSE.append(utils.mse(i1_np, res[i]))

    if min(valueComp) >= valueComp[-1] :
        pictComp = res[i]
    if min(valueMSE) >= valueMSE[-1] :
        pictMSE = res[i]

    i+= 1

#print min(value)

cv2.imshow("result", pictComp)
cv2.imshow("resultMSE", pictMSE)
print (utils.compareImages(i1, Image.fromarray(pictComp, 'RGB')))
print (utils.mse(i1_np, pictMSE))
cv2.imshow("origin", utils.cropIris(origin, crop_array['x1'],crop_array['y1'],crop_array['x2'],crop_array['y2']))

############################cv2.inRange()

cv2.waitKey(0)
cv2.destroyAllWindows()

