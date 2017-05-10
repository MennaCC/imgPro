#!/usr/bin/python
import cv2.cv as cv 
import cv2
import math
import imutils
import numpy as np
import os

r = 0
class  Cropper():
	
	def __init__(self, img, imgName):
        self.image = img
        self.name = imgName
        self._crop()

    def _crop(self):

	    img = cv2.GaussianBlur(self.image, (5,5), 0)
	    circles = cv2.HoughCircles(img, cv.CV_HOUGH_GRADIENT,1,20,
		                            param1=50,param2=30,minRadius=35,maxRadius=50)
	    circles = np.uint16(np.around(circles))
	 
	    deltax = int(circles[0, :][0][0]) - int(circles[0, :][1][0])
	    deltay = int(circles[0, :][0][1]) - int(circles[0, :][1][1])
	    theta = math.atan((1.0*deltay)/deltax)
	    theta = theta * 180 / math.pi

	    rows,cols = self.image.shape
	    M = cv2.getRotationMatrix2D((cols/2,rows/2) , theta ,1)
	    rotated = cv2.warpAffine(self.image,M,(cols,rows))
	    # rotated = imutils.rotate(rotated, angle=theta)
	    resized = cv2.resize(rotated, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	    height = 1700 
	    width = 1200
	    resized = cv2.resize(rotated, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)

	    # cropped = resized[1480:2700, 100:2080]
	    crop_img = []
	    crop_img.append(resized[1480:2700, 200:850])
	    name1 = "crop1_" + str(r) + ".png"
	    cv2.imwrite(name1, crop_img[0])
	    crop_img.append(resized[1480:2700, 850:1500])
	    name2 = "crop2_" + str(r) + ".png"
	    cv2.imwrite(name2, crop_img[1])
	    crop_img.append(resized[1480:2700, 1500:2200])
	    name3 = "crop3_" + str(r) + ".png"
	    cv2.imwrite(name3, crop_img[2])
	    r += 1
	    return crop_img


	
