from PIL import Image
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import time
# Open the image form working directory
import cv2
import numexpr as ne

import signal
import sys


#Initiate camera
videoCaptureObject = cv2.VideoCapture(0)
i=2

while(i>1):
    #Take a picture.
    ret,frame = videoCaptureObject.read()
    #Save the picture in the current directory
    #OverWrite it. Temporary solution.
    cv2.imwrite("NewPicture.jpg",frame)
    #Add delay for temp solution
    time.sleep(0.5)
    image = cv2.imread('NewPicture.jpg')
    #Take image in numpy array
    data = np.array(image)
    a2D = data.reshape(-1,data.shape[-1])
    #R G and B values of the image captured
    r = a2D[:,0]
    g = a2D[:,1]
    b = a2D[:,2]

    #print(a2D)
    pixels = np.float32(image.reshape(-1, 3))

    #Use Kmeans to find the dominant colors in the picture
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    
    #Pick the RGB color with the highest count (the dominant)

    dominant = palette[np.argmax(counts)]
    print(dominant)

    #Uncomment to show the color in matplotlib
    #plt.imshow([[[int(dominant[0]),int(dominant[1]),int(dominant[2])]]])
    #plt.show()



def signal_handler(signal, frame):
    videoCaptureObject.release()
    cv2.destroyAllWindows()
    sys.exit(0)
