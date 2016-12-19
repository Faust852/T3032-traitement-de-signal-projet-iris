import numpy
import cv2
import utils
import Image
import glob
import argparse
from scipy import ndimage
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())


imageFolderPath = '/home/faust/PycharmProjects/T3032-traitement-de-signal-projet-iris/projectiris/images/'
imagePath = glob.glob(imageFolderPath+'/NIR_2/*.bmp')
im_array = numpy.array( [numpy.array(Image.open(imagePath[i]).convert('L'), 'f') for i in range(len(imagePath))] )

i1,s1,n1,b1= utils.wrapperIris(args['image'])

for picture in glob.glob('/home/faust/PycharmProjects/T3032-traitement-de-signal-projet-iris/projectiris/images/NIR_2/*.bmp') :
    i,s,n,b = utils.wrapperIris(picture)
    v1 = utils.comparePattern(s1, s)
    if (v1 > 199):
        print picture
