#!/usr/bin/python3
import cv2
import imutils

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

image = cv2.imread('1.png')
# image = cv2.imread('1Uncropped.png')

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (7, 7), 0)
bilateralblur = cv2.bilateralFilter(image,9,75,75)

edged = cv2.Canny(blurred, 75, 200)
bilateraledged = cv2.Canny(bilateralblur,75,200)

thresh = cv2.threshold(bilateraledged, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

thresh_edged = cv2.threshold(edged, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh_edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
print(cnts)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
print("after")
print(cnts)
questionCnts = []

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour, then use the
	# bounding box to derive the aspect ratio
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	print(c)
	print(x,y,w,h)

	# in order to label the contour as a question, region
	# should be sufficiently wide, sufficiently tall, and
	# have an aspect ratio approximately equal to 1
	if w >= 10 and h >= 10 and ar >= 0.9 and ar <= 1.1:
		questionCnts.append(c)

print(questionCnts)


# cv2.imshow("gray",gray)
cv2.imwrite("resultBlurredCroppped7.jpg",blurred)
cv2.imwrite("resultedgedCropped7.jpg",edged)
cv2.imwrite("resultBilateralBlurred97575.jpg",bilateralblur)
cv2.imwrite("resultBilateralEdged97575.jpg", bilateraledged)
cv2.imwrite("resultBilateralEdged97575_thresh.jpg", thresh)
cv2.imwrite("resultBilateralEdged97575_threshEdged.jpg", thresh_edged)




