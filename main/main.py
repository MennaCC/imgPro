#!/usr/bin/python
# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import math

ANSWER_KEY = {0: 2, 1: 1, 2: 2, 3: 3, 4: 4}

input_image = cv2.imread('1.png')

gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
bilateralblur = cv2.bilateralFilter(gray, 9, 75, 75)

edged = cv2.Canny(input_image, 75, 200)
bilateraledged = cv2.Canny(bilateralblur, 75, 200)

thresh = cv2.threshold(blurred, 0, 255,
                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

thresh_edged = cv2.threshold(bilateralblur, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

resImage = edged + (255 - thresh)
cv2.imwrite("resImage.png", resImage)

# remove everything but the circles
image = cv2.imread("resImage.png")
mask = np.ones(image.shape[:2], dtype="uint8") * 255
img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                     3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation

cv2.imwrite('ndefa.jpg', dilated)

contourrs, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours
index = 0
questionCnts = []

for contour in contourrs:
    # get rectangle bounding contour
    [x, y, w, h] = cv2.boundingRect(contour)
    ar = w / float(h)
    # in order to label the contour as a question, region
    # should be sufficiently wide, sufficiently tall, and
    # have an aspect ratio approximately equal to 1
    if w >= 3 and h >= 3 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(contour)
    # Don't plot small false positives that aren't text
    if h > 500:
        continue
    # draw rectangle around contour on original image
    # cv2.rectangle(ima,(x,y),(x+w,y+h),(255,0,255),2)
    cv2.drawContours(mask, [contour], -1, 0, -1)

ima = cv2.bitwise_and(image, image, mask=mask)
cv2.imwrite('ndefa.jpg', ima)

###########################################

# sort the question contours top-to-bottom, then initialize
# the total number of correct answers
questionCnts = contours.sort_contours(questionCnts,
                                      method="top-to-bottom")[0]

# get rid of the contours of the characters
# questionCnts = questionCnts[3:]

cv2.drawContours(image, questionCnts, -1, (255, 255, 255), 3)
cv2.imwrite("cont.jpg", image)



# correct = 0
# # each question has 4 possible answers, to loop over the
# # question in batches of 5
# for (q, i) in enumerate(np.arange(0, 20, 4)):
#     # sort the contours for the current question from
#     # left to right, then initialize the index of the
#     # bubbled answer
#     cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
#     bubbled = None
#     # cv2.drawContours(resImage, cnts, -1, (255,0,0), 3)
#     # cv2.imwrite("cont.jpg", resImage )
#     # cv2.imshow("aaa",resImage)
#     # cv2.waitKey(0)

#     # loop over the sorted contours
#     for (j, c) in enumerate(cnts):
#         # construct a mask that reveals only the current
#         # "bubble" for the question
#         mask = np.zeros(thresh.shape, dtype="uint8")
#         cv2.drawContours(mask, [c], -1, 255, -1)

#             # apply the mask to the thresholded image, then
#             # count the number of non-zero pixels in the
#             # bubble area
#         mask = cv2.bitwise_and(thresh, thresh, mask=mask)
#         total = cv2.countNonZero(mask)

#             # if the current total has a larger number of total
#             # non-zero pixels, then we are examining the currently
#             # bubbled-in answer
#         if bubbled is None or total > bubbled[0]:
#             bubbled = (total, j)
#             # initialize the contour color and the index of the
#             # *correct* answer
#     color = (0, 0, 255)
#     k = ANSWER_KEY[q]
#     print(k, bubbled[1])
#     # check to see if the bubbled answer is correct
#     if k == bubbled[1]:
#         color = (0, 255, 0)
#         correct += 1
# print(correct)