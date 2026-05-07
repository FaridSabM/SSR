#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.optimize import linear_sum_assignment 


# In[2]:


# this function Permute labels of l2 to match l1 as much as possible and use for ACC evaluation
def best_map(l1, l2):
    """
    Permute labels of l2 to match l1 as much as possible
    """
    if len(l1) != len(l2):
        print("L1.shape must == L2.shape")
        exit(0)

    label1 = np.unique(l1)
    n_class1 = len(label1)

    label2 = np.unique(l2)
    n_class2 = len(label2)

    n_class = max(n_class1, n_class2)
    G = np.zeros((n_class, n_class))

    for i in range(0, n_class1):
        for j in range(0, n_class2):
            ss = l1 == label1[i]
            tt = l2 == label2[j]
            G[i, j] = np.count_nonzero(ss & tt)
    row_indices, col_indices = linear_sum_assignment(-G)
    A = list(zip(row_indices, col_indices))
    #A = la.linear_assignment(-G)

    new_l2 = np.zeros(l2.shape)
    for i in range(0, n_class2):
        new_l2[l2 == label2[A[i][1]]] = label1[A[i][0]]
    return new_l2.astype(int)


# In[ ]:




