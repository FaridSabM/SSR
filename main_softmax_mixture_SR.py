#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
%% Softmax-Mixture-SR: Softmax-based Mixture Self-Representation for Unsupervised Feature Selection
%% Farid Saberi-Movahed
%% Email: f.saberimovahed@kgut.ac.ir; fdsaberi@gmail.com

%% Optimization problem:
%% min_{A,B} ||X - XB||_F^2 + alpha * ||X - AX||_F^2 + beta * ||B||_{2,1}
%%
%% subject to:
%% B = Softmax_row(Z),
%% A = Softmax_column(Y)
%%
%% in which
% X        : (m × n) data matrix, with m samples and n features
% A        : (m × m) sample-wise self-representation coefficient matrix
% B        : (n × n) feature-wise self-representation coefficient matrix
% Z        : latent variable for row-wise softmax mapping of B
% Y        : latent variable for column-wise softmax mapping of A
% alpha    : regularization parameter for sample reconstruction
% beta     : regularization parameter for the l2,1 norm of B
% lr       : learning rate for updating Z and Y
%
% Feature ranking is obtained from the l2-norm of columns of B.
% The top-k ranked features are selected.

%% Required function:
%% [A, B, X_selected] = softmax_mixture_SR(X, alpha, beta, lr, max_iter, k, tol, eps)
%%
%% Inputs:
%   X          : (m × n) data matrix
%   alpha      : regularization parameter for A
%   beta       : regularization parameter for B
%   lr         : learning rate
%   max_iter   : maximum number of iterations
%   k          : number of selected features
%   tol        : convergence tolerance
%   eps        : small constant to avoid division by zero
%%
%% Outputs:
%   A          : (m × m) learned sample-wise coefficient matrix
%   B          : (n × n) learned feature-wise coefficient matrix
%   X_selected : (m × k) data matrix with selected features

%% Required packages:
%% numpy
%% scipy
%% scikit-learn
%% functions.Bestmap
%% functions.Purity
%% sr_methods.Softmax_Mixture_SR
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.io import loadmat

from functions.Bestmap import best_map
from functions.Purity import purity_score
from sr_methods.Softmax_Mixture_SR import softmax_mixture_SR


def run_softmax_mixture_sr():
    np.random.seed(0)

    NameMethod = 'Softmax-Mixture-SR'

    # -------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------
    # JAFFE dataset is loaded manually from file.
    # The file is assumed to be stored in:
    # JAFFE.mat
    #
    # Required arrays inside the file:
    #   fea : data matrix of shape (m, n)
    #   gnd : ground-truth labels of shape (m,)
    #
    # If your file has different key names, replace 'fea' and 'gnd' below.

    data = loadmat('JAFFE.mat')
    X = data['fea']
    Y = data['gnd'].ravel()

    Namedata = 'JAFFE'
    No_class = len(np.unique(Y))

    # -------------------------------------------------------------
    # Initial parameters
    # -------------------------------------------------------------
    max_iter = 5       # Suggested range {5, 10, 20, 30}
    k = 90             # Suggested range {10, 20, ..., 100}
    alpha = 1e-3       # Suggested range {1e-6, 1e-3, 1, 1e3, 1e6}
    beta = 1e-3        # Suggested range {1e-6, 1e-3, 1, 1e3, 1e6}
    lr = 1e-3          # Suggested range {1e-6, 1e-3, 1, 1e3, 1e6}
    tol = 1e-6
    eps = 1e-8

    # -------------------------------------------------------------
    # Proposed method
    # -------------------------------------------------------------
    A, B, X_selected = softmax_mixture_SR(
        X, alpha, beta, lr, max_iter, k, tol=tol, eps=eps
    )

    # -------------------------------------------------------------
    # Clustering phase
    # -------------------------------------------------------------
    tempNMI = np.zeros(20)
    tempACC = np.zeros(20)
    tempPURITY = np.zeros(20)

    for i in range(20):
        kmeans = KMeans(n_clusters=No_class, n_init=100, random_state=i)
        y_pred = kmeans.fit_predict(X_selected)

        tempNMI[i] = normalized_mutual_info_score(Y, y_pred)
        y_permuted = best_map(Y, y_pred)
        tempACC[i] = accuracy_score(Y, y_permuted)
        tempPURITY[i] = purity_score(Y, y_pred)

    # -------------------------------------------------------------
    # Final results
    # -------------------------------------------------------------
    ACC = np.mean(tempACC) * 100
    NMI = np.mean(tempNMI) * 100
    PURITY = np.mean(tempPURITY) * 100

    print('Method Name:', NameMethod)
    print('Dataset Name:', Namedata)
    print('alpha =', alpha)
    print('beta =', beta)
    print('lr =', lr)
    print('k =', k)
    print('max_iter =', max_iter)
    print('----------------------------------------')
    print('ACC = {:.4f}'.format(ACC))
    print('NMI = {:.4f}'.format(NMI))
    print('PURITY = {:.4f}'.format(PURITY))


if __name__ == "__main__":
    run_softmax_mixture_sr()


# In[ ]:




