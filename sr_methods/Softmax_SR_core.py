#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.preprocessing import normalize
from sr_methods.softmax_utils import softmax_row, softmax_jacobian_row


def softmax_SR_m(X, alpha, lr, max_iter, k, tol=1e-6, eps=1e-8):
    """
    Softmax-based Self-Representation (softmax-SR) for unsupervised feature selection.

    Minimizes:
        L = ||X - X B||_F^2 + alpha * ||B||_{2,1}
    with B = Softmax(Z).

    Notes
    -----
    The algorithm alternates between:
    (1) multiplicative updates for B, and
    (2) gradient-based updates for Z through the softmax mapping.
    The diagonal matrix G is derived from the l2,1 norm. In this implementation:
        G_ii = 1/2 * ||B_i||_2

    Parameters
    ----------
    X : ndarray of shape (m, n)
        Data matrix with m samples and n features.
    alpha : float
        Regularization parameter for the l2,1 norm.
    lr : float
        Learning rate for updating Z.
    max_iter : int
        Maximum number of iterations.
    k : int
        Number of features to select.
    tol : float, default=1e-6
        Convergence tolerance (not used in the current implementation).
    eps : float, default=1e-8
        Small constant to avoid division by zero.

    Returns
    -------
    B : ndarray of shape (n, n)
        Self-representation coefficient matrix after softmax mapping.
    X_selected : ndarray of shape (m, k)
        Data matrix restricted to the top-k selected features.
    """
    m, n = X.shape

    # Initialization
    Z = np.random.randn(n, n)
    B = softmax_row(Z)
    G = np.eye(n)
    XTX = X.T @ X

    for _ in range(max_iter):
        # Update B (multiplicative update)
        num_B = XTX
        den_B = XTX @ B + alpha * (G @ B)
        B *= num_B / np.maximum(den_B, eps)

        # Update G (diagonal)
        for i in range(n):
            G[i, i] = 0.5 * np.maximum(np.linalg.norm(B[i, :]), eps)

        # Gradient step on Z through softmax
        grad_Z = np.zeros_like(Z)
        for i in range(n):
            J = softmax_jacobian_row(B[i, :])
            grad_Z[i, :] = J.T @ B[i, :]
        Z -= lr * grad_Z

    # Final softmax mapping
    B = softmax_row(Z)

    # Feature ranking
    scores = np.linalg.norm(B, axis=1)
    ranked_idx = np.argsort(-scores)
    X_selected = X[:, ranked_idx[:k]]
    X_selected = normalize(X_selected, axis=1)  # normalize features column-wise

    return B, X_selected


# In[ ]:




