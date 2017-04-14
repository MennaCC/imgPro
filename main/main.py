#!/usr/bin/python
# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

image = cv2.imread('1.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (7, 7), 0)
bilateralblur = cv2.bilateralFilter(gray,9,75,75)

edged = cv2.Canny(image, 75, 200)
bilateraledged = cv2.Canny(bilateralblur,75,200)

thresh = cv2.threshold(blurred, 0, 255,
	cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

thresh_edged = cv2.threshold(bilateralblur, 0, 255,
	cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

resImage = edged + (255-thresh)


cnts = cv2.findContours(resImage.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[0] if imutils.is_cv2() else cnts[1]
questionCnts = []


cv2.drawContours(resImage, cnts, -1, (128,255,0), 3)
cv2.imshow("resImage", resImage)
cv2.waitKey(0)

cv2.imwrite('resImage.jpg', resImage)
# loop over the contours
for c in cnts:
	# compute the bounding box of the contour, then use the
	# bounding box to derive the aspect ratio
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	# in order to label the contour as a question, region
	# should be sufficiently wide, sufficiently tall, and
	# have an aspect ratio approximately equal to 1
	if w >= 3 and h >= 3 and ar >= 0.9 and ar <= 1.1:
		questionCnts.append(c)

# sort the question contours top-to-bottom, then initialize
# the total number of correct answers
questionCnts = contours.sort_contours(questionCnts,
    method="top-to-bottom")[0]
# get rid of the contours of the characters 
questionCnts = questionCnts[12:]





