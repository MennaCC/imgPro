#!/usr/bin/python
# import the necessary packages
# import pandas as pandas
from imutils import contours
import numpy as np
import cv2
import os
from crop import Cropper

ANSWER_KEY = {0: 1, 1:0 , 2: 1, 3: 2, 4: 3, 5:1, 6:0, 7:1, 8:3, 9:0, 10:2, 11:2}

# ANSWER_KEY = {0: 1, 1:0 , 2: 1, 3: 2, 4: 3, 5:1, 6:0, 7:1, 8:3, 9:0, 10:2, 11:2,
# 	12:3, 13:1, 14:1, 15:0, 16:3, 17:2, 18:2, 19:2, 20:1, 21:2, 22:3, 23:2, 24:0, 
# 	25:1, 26:2, 27:2, 28:3, 29:0, 30:0, 31:2, 32:1, 33:1, 34:3, 35:1, 36:2, 37:3, 
# 	38:2, 39:2, 40:1, 41:2, 42:1, 43:2, 44:1}

WHICH_CV = None

class Image:
    def __init__(self, img):
        self.originalImage      = img            #cv2 image object > the one we get from imread

        self.croppedImage       = self._crop()
        self.show("cropped")

        self.preprocessedImage  = self._preProcess()
        self.show("preprocessed")

        self.resultImage        = self._grade()

        self.name = "1"

    def get_contours(self,image):
        if WHICH_CV == "shu":
            contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else :
            _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return  contours


    def _preProcess(seelf):
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


        contours = self.get_contours(dilated)
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

        cons = self.get_contours(thresh_edged)
        questionCnts = []

        for contour in cons:
            (x, y, w, h) = cv2.boundingRect(contour)
            ar = w / float(h)
            if w >= 3 and h >= 3 and ar >= 0.8 and ar <= 1.2:
                questionCnts.append(contour)

        questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]

        correct = 0
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
        cropper = Cropper(self.originalImage)
        self.croppedImage = cropper.get_cropped_img()
        # self.croppedImage = cv2.imread('crop.jpg')
        self.show('cropped')

        return self.croppedImage


    #mmkn a3mlo enum bs mlee4 mzag dlw2ty
    def show(self,whichImage):
        self.namewindow = ""
        if whichImage == "original":
            im = self.originalImage
            self.namewindow = "original image"
            cv2.namedWindow("original image", cv2.WINDOW_NORMAL)

        elif whichImage == "cropped":
            im = self.croppedImage
            self.namewindow = "cropped image"
            cv2.namedWindow("cropped image", cv2.WINDOW_NORMAL)

        elif whichImage == "preprocessed":
            im = self.preprocessedImage
            self.namewindow = "preprocessed image"
            cv2.namedWindow("preprocessed image", cv2.WINDOW_NORMAL)

        elif whichImage == "result":
            im = self.resultImage
            self.namewindow = "result image"
            cv2.namedWindow("result image", cv2.WINDOW_NORMAL)

        cv2.imshow(self.namewindow, im)
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
            grades_dict[im.name] = im._grade()

    def set_AnswerKey(self, dict):
        self.ANSWER_KEY = dict

    # def evaluate(self, obtained_marks):
    # 	# read the csv file
		# marks_file = pandas.read_csv('/home/shimaa/Desktop/Working_Space/College/ImageProcessing/Project/Materials/train.csv')
		# #cut only the filename and the mark columns
		# marks_file = marks_file[['FileName','Mark']]
		# # convert the dataframe to dictionary
		# correct_marks = marks_file.set_index('FileName')['Mark'].to_dict()
		# #compare the results
		# for key in obtained_marks :
    #         if key in correct_marks:
    #             if correct_marks[key] != obtained_marks[key] :
    #         		print("For the file {}".format(key))
    #         		print("Correct mark is {} but the obtained is {}".format(correct_marks[key], obtained_marks[key]))

if __name__ == '__main__':
    # grader = Grader('/home/bubbles/3anQa2/College/imgpro/train')
    grader = Grader('./hi')















