
# coding: utf-8

# In[1]:

def awel_wa7ed(image):
    img = cv2.imread(image)
    crop_img = img[1500:2700, 200:850]
    cv2.imwrite("crop1.jpg",crop_img)


# In[2]:

def tany_wa7ed(image):
    img = cv2.imread(image)
    crop_img = img[1500:2700, 850:1500]
    cv2.imwrite("crop2.jpg",crop_img)


# In[ ]:

def talet_wa7ed(image):
    img = cv2.imread(image)
    crop_img = img[1500:2700, 1500:2200]
    cv2.imwrite("crop3.jpg",crop_img)

