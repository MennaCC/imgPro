
# coding: utf-8

# In[6]:

from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


# In[7]:

image_path='Desktop/S_12_hppscan123.png'


# In[8]:

img = cv2.imread(image_path)
crop_img = img[650:1753, 10:1240]


# In[9]:

image = crop_img
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)


# In[10]:

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
docCnt = None


# In[11]:

if len(cnts) > 0:

	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True) 
		if len(approx) == 4:
			docCnt = approx
			break


# In[12]:

cv2.drawContours(image, [docCnt], -1, (0, 255, 0), 3)


# In[13]:

paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))


# In[ ]:

cv2.namedWindow("elContour el gamel", cv2.WINDOW_NORMAL)    
imS = cv2.resize(paper, (600, 600))                    
cv2.imshow("elContour el gamel", imS)                            
cv2.waitKey(0)  


# In[ ]:



