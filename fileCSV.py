
# coding: utf-8

# In[20]:

import csv 


# In[21]:

index_path='Desktop/file.csv'


# In[22]:

r={'swra1':25,'swra2':26,'swra5':6,'swra6':105,'swra9':11}


# In[23]:

with open (index_path,'wb') as csv_file:
    writer=csv.writer(csv_file)
    for key,value in r.items():
        writer.writerow([key,value])


# In[ ]:




# In[ ]:




# In[ ]:



