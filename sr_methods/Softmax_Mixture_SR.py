#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.preprocessing import normalize
from sr_methods.softmax_utils import softmax_row, softmax_jacobian_row, softmax_column, softmax_jacobian_column

def softmax_mixture_SR(X, alpha, beta, lr, max_iter, k, tol=1e-6, eps=1e-8):
    """
    Softmax-based Mixture Self-Representation (softmax-mixture SR) for
    unsupervised feature selection.

    Minimizes:
        L = ||X - X B||_F^2 + alpha * ||X - A X||_F^2 + beta * ||B||_{2,1}
    with B = Softmax_row(Z), A = Softmax_column(Y).

    Notes
    -----
    The algorithm alternates between:
    (1) multiplicative updates for A and B,
    (2) updates of the diagonal matrix G derived from the l2,1 norm, and
    (3) gradient-based updates for Z and Y through their respective softmax maps.
    In this implementation, G is updated as:
        G_ii = 1/2 * ||B_i||_2

    Parameters
    ----------
    X : ndarray of shape (m, n)
        Data matrix with m samples and n features.
    alpha : float
        Regularization parameter for the reconstruction term of A.
    beta : float
        Regularization parameter for the l2,1 norm of B.
    lr : float
        Learning rate for updating Z and Y.
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
    A : ndarray of shape (m, m)
        Sample-wise self-representation coefficient matrix.
    B : ndarray of shape (n, n)
        Feature-wise self-representation coefficient matrix after softmax mapping.
    X_selected : ndarray of shape (m, k)
        Data matrix restricted to the top-k selected features.
    """
    m, n = X.shape

    # Initialization
    G = np.eye(n)
    np.random.seed(10)
    Z = np.random.rand(n, n)
    Y = np.random.rand(m, m)
    A = softmax_column(Y)
    B = softmax_row(Z)

    XTX = X.T @ X
    XXT = X @ X.T

    for _ in range(max_iter):
        # Update A (multiplicative update)
        num_A = alpha * XXT
        den_A = alpha * (A @ XXT)
        A *= num_A / np.maximum(den_A, eps)

        # Update B (multiplicative update)
        num_B = XTX
        den_B = XTX @ B + beta * (G @ B)
        B *= num_B / np.maximum(den_B, eps)

        # Update G (diagonal)
        for i in range(n):
            G[i, i] = 0.5 * np.maximum(np.linalg.norm(B[i, :]), eps)

        # Gradient step on Z through softmax (row-wise)
        grad_Z = np.zeros_like(Z)
        for i in range(n):
            J = softmax_jacobian_row(B[i, :])
            grad_Z[i, :] = J.T @ B[i, :]
        Z -= lr * grad_Z

        # Gradient step on Y through softmax (column-wise)
        grad_Y = np.zeros_like(Y)
        for j in range(m):
            J = softmax_jacobian_column(A[:, j])
            grad_Y[:, j] = J.T @ A[:, j]
        Y -= lr * grad_Y

    # Final softmax mapping
    B = softmax_row(Z)
    A = softmax_column(Y)

    # Feature ranking
    scores = np.linalg.norm(B, axis=0)
    ranked_idx = np.argsort(-scores)
    X_selected = X[:, ranked_idx[:k]]
    X_selected = normalize(X_selected, axis=1)  # normalize features column-wise

    return A, B, X_selected


# In[ ]:




