
# coding: utf-8

# In[1]:

import cv2, os 
import numpy as np 
from PIL import Image 


# In[2]:

def get_images(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    for i in image_paths:
        img = cv2.imread(i,0)
      
        cv2.imshow("swer", img)
        cv2.waitKey(1000)
    return images;



# In[3]:

path ='Desktop/imgpro'
images = get_images(path)


# In[ ]:



