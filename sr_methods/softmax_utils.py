#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def softmax_row(Z, axis=1):
    """Compute softmax over rows (default axis=1) with numerical stability."""
    Z = Z - np.max(Z, axis=axis, keepdims=True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=axis, keepdims=True)


# In[3]:


def softmax_column(Z, axis=0):
    """Compute softmax over columns (default axis=0) with numerical stability."""
    Z = Z - np.max(Z, axis=axis, keepdims=True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=axis, keepdims=True)


# In[4]:


def softmax_jacobian_row(h):
    """Return the Jacobian matrix for a single softmax ROW vector."""
    return np.diag(h) - np.dot(h.T, h)


# In[5]:


def softmax_jacobian_column(h):
    """Return the Jacobian matrix for a single softmax COLUMN vector."""
    return np.diag(h) - np.outer(h, h)


# In[ ]:




