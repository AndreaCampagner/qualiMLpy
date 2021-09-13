import numpy as np
from sklearn.metrics import pairwise_kernels

def mmd(X_train, X_test, kernel='sigmoid'):
    intra = np.sum(pairwise_kernels(X_train, X_train, metric=kernel))
    intra += np.sum(pairwise_kernels(X_test, X_test, metric=kernel))
    extra = np.sum(pairwise_kernels(X_train, X_test, metric=kernel))
    return intra - 2*extra