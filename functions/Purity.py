#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn import metrics


# In[2]:


#This function is used for  compute purity evaluation
def purity_score(y_true, y_pred):
    # Compute confusion matrix
    confusion_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # Return purity
    return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)


# In[ ]:




