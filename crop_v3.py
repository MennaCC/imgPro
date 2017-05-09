
# coding: utf-8

# In[239]:

from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import math
from decimal import *
import cv2, os 
import numpy as np 
from PIL import Image 
 


# In[240]:

def theta_hanem (d1,d2):
    
    theta= math.degrees(math.atan(Decimal(d2)/Decimal(d1)))
    return theta 


# In[218]:

def awel_wa7ed(image):
    img = cv2.imread(image)
    crop_img = img[1500:2700, 200:850]
    cv2.imwrite("crop1.jpg",crop_img)


# In[219]:

def tany_wa7ed(image):
    img = cv2.imread(image)
    crop_img = img[1500:2700, 850:1500]
    cv2.imwrite("crop2.jpg",crop_img)


# In[220]:

def talet_wa7ed(image):
    img = cv2.imread(image)
    crop_img = img[1500:2700, 1500:2200]
    cv2.imwrite("crop3.jpg",crop_img)


# In[221]:

def rotate (image,theta):
    img = cv2.imread(image,0)
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite("rotate.jpg",dst)


# In[222]:

def crop (image):
    img = cv2.imread(image)
    crop_img = img[1500:2700, 200:2080]
    cv2.imwrite("crop.jpg",crop_img)


# In[223]:

image_path1='Desktop/Untitled Folder 3/S_1_hppscan18.png'


# In[224]:

image__ = cv2.imread(image_path1)


# In[225]:

res = cv2.resize(image__,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
height=1700
width=1200
res = cv2.resize(image__,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)


# In[226]:

cv2.imwrite("a1.jpg",res)


# In[227]:

image2='a1.jpg'


# In[228]:

image__=cv2.imread(image2)


# In[229]:

gray = cv2.cvtColor(image__, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(blurred,10,200)


# In[230]:

cv2.imwrite("aedged.jpg",edged)


# In[231]:

image1="aedged.jpg"
image=cv2.imread(image1)


# In[232]:

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
after_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 29, 5)
kernel = np.ones((5,5),np.uint8)
after_thresh = cv2.morphologyEx(after_thresh, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("155.png",after_thresh)


# In[233]:

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


# In[234]:

image_='155.png'
image = cv2.imread(image_)


# In[235]:

resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite("1ape.jpg",thresh)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
sd = ShapeDetector()


# In[236]:

circles=[]
j=0
for c in cnts:
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c) 
    
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        if(shape=="circle" and cY>1000 and cv2.contourArea(c)>2000 ):
            circles.append(cX)
            circles.append(cY)
            area = cv2.contourArea(c)
            print shape,cX,cY
            print area
                    
	cv2.putText(image, shape, (cX,cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (0, 255,255), 2)
    
cv2.imwrite("1detect.jpg",image)


# In[237]:

d1=circles[0]-circles[2] 
d2=circles[1]-circles[3]


# In[238]:

theta1=theta_hanem(d1,d2)
rotate(image2,theta1)
crop('rotate.jpg')
awel_wa7ed('rotate.jpg')
tany_wa7ed('rotate.jpg')
talet_wa7ed('rotate.jpg')


# In[ ]:




# In[ ]:




# In[55]:




# In[ ]:




# In[ ]:



