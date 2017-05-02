#!/usr/bin/python
# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import numpy as np
import cv2
import os

ANSWER_KEY = {0: 1, 1:0 , 2: 1, 3: 2, 4: 3, 5:1, 6:0, 7:1, 8:3, 9:0, 10:2, 11:2}

class Image:
    def __init__(self, img):
        self.originalImage      = img            #cv2 image object > the one we get from imread

        self.croppedImage       = self._crop()
        self.preprocessedImage  = self._preProcess()
        self.resultImage        = self._grade()
        self.name = "1"

    def _preProcess(self):
        input_image = self.croppedImage

        gray        = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        blurred     = cv2.GaussianBlur(gray, (7, 7), 0)
        edged       = cv2.Canny(input_image, 75, 200)
        thresh      = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        resImage    = edged + (255 - thresh)
        cv2.imwrite("resImage.png", resImage)

        # remove everything but the circles
        image           = cv2.imread("resImage.png")
        img2gray        = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, mask       = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
        image_final     = cv2.bitwise_and(img2gray, img2gray, mask=mask)
        # for black text , cv.THRESH_BINARY_INV
        ret, new_img    = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)
        # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
        kernel          = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dilated         = cv2.dilate(new_img, kernel, iterations=9)

        
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if h > 500:
                continue
            cv2.drawContours(mask, [contour], -1, 0, -1)

        ima = cv2.bitwise_and(image, image, mask=mask)
        cv2.imwrite('ndefa.jpg', ima)

        kernel = np.ones((4, 4), np.uint8)
        dilation = cv2.dilate(ima, kernel, iterations=1)
        cv2.imwrite('dilated.jpg', dilation)

        return dilation


    def _grade(self):
        gray = cv2.cvtColor(self.preprocessedImage, cv2.COLOR_BGR2GRAY)
        bilateralblur = cv2.bilateralFilter(gray, 9, 75, 75)

        thresh_edged = cv2.threshold(bilateralblur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        cons, hierarchy = cv2.findContours(thresh_edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours
        questionCnts = []

        for contour in cons:
            (x, y, w, h) = cv2.boundingRect(contour)
            ar = w / float(h)
            if w >= 3 and h >= 3 and ar >= 0.8 and ar <= 1.2:
                questionCnts.append(contour)
                print contour

        questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]

        correct = 0
        self.show("cropped")
        for (q, i) in enumerate(np.arange(0, 44, 4)):
            cnts = contours.sort_contours(questionCnts[i:i + 4], method="left-to-right")[0]
            bubbled = None
            for (j, c) in enumerate(cnts):
                mask = np.zeros(thresh_edged.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)

                mask = cv2.bitwise_and(thresh_edged, thresh_edged, mask=mask)
                total = cv2.countNonZero(mask)
                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, j)

            k = ANSWER_KEY[q]
            print(q, k, bubbled[1])
            if k == bubbled[1]:
                correct += 1
        print(correct)

    def _crop(self):
        image = self.originalImage[650:1753, 10:1240]


        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        docCnt = None

        if len(cnts) > 0:

            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    docCnt = approx
                    break

        cv2.drawContours(image, docCnt, -1, (0, 255, 0), 3)

        paper = four_point_transform(image, docCnt.reshape(4, 2))

        imS = cv2.resize(paper, (600, 600))
        self.preprocessedImage = imS

        return imS


    #mmkn a3mlo enum bs mlee4 mzag dlw2ty
    def show(self,whichImage):
        if whichImage == "original":
            im = self.originalImage
            cv2.namedWindow("original image", cv2.WINDOW_NORMAL)

        elif whichImage == "cropped":
            im = self.preprocessedImage
            cv2.namedWindow("cropped image", cv2.WINDOW_NORMAL)

        elif whichImage == "result":
            im = self.resultImage
            cv2.namedWindow("result image", cv2.WINDOW_NORMAL)

        cv2.imshow("img", im)
        cv2.waitKey(0)


class Grader:
    def __init__(self, path):
        self.images = []

        self.get_images(path)
        self.iterate_images()



    def get_images(self,path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]

        for i in image_paths:
            img = cv2.imread(i)
            self.images.append(img)

        return self.images

    def iterate_images(self):
        grades_dict = {}
        for image in self.images:
            im = Image(image)
            grades_dict[im.get_name()] = im.get_grade()

    def set_AnswerKey(self, dict):
        self.ANSWER_KEY = dict

if __name__ == '__main__':
    grader = Grader('/home/shimaa/Desktop/Working_Space/College/ImageProcessing/Materials/tr')















