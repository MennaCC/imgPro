import numpy as np
import imutils
import cv2
import math
from decimal import *

class Cropper():
    def __init__(self, img):
        self.image__ = img
        # self.image__ = cv2.imread(image_path1)

        res = cv2.resize(self.image__, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        height = 1700
        width = 1200
        res = cv2.resize(self.image__, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite("a1.jpg", res)

        image2 = 'a1.jpg'

        self.image__ = cv2.imread(image2)

        gray = cv2.cvtColor(self.image__, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 10, 200)

        cv2.imwrite("aedged.jpg", edged)

        image1 = "aedged.jpg"
        image = cv2.imread(image1)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
        after_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 29, 5)
        kernel = np.ones((5, 5), np.uint8)
        after_thresh = cv2.morphologyEx(after_thresh, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("155.png", after_thresh)

        image_ = '155.png'
        image = cv2.imread(image_)

        resized = imutils.resize(image, width=300)
        ratio = image.shape[0] / float(resized.shape[0])
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite("1ape.jpg", thresh)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        sd = ShapeDetector()

        circles = []
        j = 0
        for c in cnts:
            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
            shape = sd.detect(c)

            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            if (shape == "circle" and cY > 1000 and cv2.contourArea(c) > 2000):
                circles.append(cX)
                circles.append(cY)
                area = cv2.contourArea(c)
                # print shape, cX, cY
                # print area

            cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 2)

        cv2.imwrite("1detect.jpg", image)

        d1 = circles[0] - circles[2]
        d2 = circles[1] - circles[3]

        theta1 = self.theta_hanem(d1, d2)
        self.rotate(image2, theta1)
        # self.crop('rotate.jpg')
        self._cutImage()

    def get_cropped_img(self):
        return self.crop_img


    def theta_hanem(self,d1, d2):
        theta = math.degrees(math.atan(Decimal(d2) / Decimal(d1)))
        return theta

    def rotate(self,image, theta):
        img = cv2.imread(image, 0)
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        cv2.imwrite("rotate.jpg", dst)

    def crop(self,image):
        img = cv2.imread('rotate.jpg')
        self.crop_img = img[1500:2700, 200:2080]
        cv2.imwrite("crop.jpg", self.crop_img)

    def _cutImage(self):
        self.crop_img = []
        img = cv2.imread('rotate.jpg')
        self.crop_img.append(img[1500:2700, 350:750])
        cv2.imwrite("crop1.jpg", self.crop_img[0])
        self.crop_img.append(img[1500:2700, 850:1500])
        cv2.imwrite("crop2.jpg", self.crop_img[1])
        self.crop_img.append(img[1500:2700, 1500:2200])
        cv2.imwrite("crop3.jpg", self.crop_img[2])


# image_path1 = '/home/bubbles/3anQa2/College/imgpro/imgPro/main/1Uncropped.png'

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        elif len(approx) == 5:
            shape = "pentagon"
        else:
            shape = "circle"
        return shape