#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.preprocessing import normalize

def SR_m(X, alpha, max_iter, k, eps=1e-8):
    """
    Softmax-based Self-Representation (SR) for unsupervised feature selection.

    Minimizes:
        L = ||X - X B||_F^2 + alpha * ||B||_{2,1}

    Notes
    -----
    Uses multiplicative updates for B and a diagonal matrix G derived from
    the l2,1 norm. G is updated as:
        G_ii = 1 / (2 * ||B_i||_2)

    Parameters
    ----------
    X : ndarray of shape (m, n)
        Data matrix with m samples and n features.
    alpha : float
        Regularization parameter for the l2,1 norm.
    max_iter : int
        Maximum number of iterations.
    k : int
        Number of features to select.
    eps : float, default=1e-8
        Small constant to avoid division by zero.

    Returns
    -------
    B : ndarray of shape (n, n)
        Self-representation coefficient matrix.
    X_selected : ndarray of shape (m, k)
        Data matrix restricted to the top-k selected features.
    """
    m, n = X.shape

    # Initialization
    B = np.random.rand(n, n)
    XTX = X.T @ X
    G = np.eye(n)

    for _ in range(max_iter):
        # Update B
        num_B = XTX
        den_B = XTX @ B + alpha * (G @ B)
        B *= num_B / np.maximum(den_B, eps)

        # Update G (diagonal)
        row_norms = np.linalg.norm(B, axis=1)
        G = np.diag(1.0 / (2.0 * np.maximum(row_norms, eps)))

    # Feature ranking
    scores = np.linalg.norm(B, axis=1)
    ranked_idx = np.argsort(-scores)
    X_selected = X[:, ranked_idx[:k]]

    # Normalize selected features column-wise
    X_selected = normalize(X_selected, axis=0)

    return B, X_selected


# In[ ]:




