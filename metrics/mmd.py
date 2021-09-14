import numpy as np
from sklearn.metrics import pairwise_kernels
from sklearn.model_selection import train_test_split

def mmd(X_a, X_b, kernel='sigmoid'):
    intra = np.sum(pairwise_kernels(X_a, X_a, metric=kernel))
    intra += np.sum(pairwise_kernels(X_b, X_b, metric=kernel))
    extra = np.sum(pairwise_kernels(X_a, X_b, metric=kernel))
    return intra - 2*extra

def mmd_test(X_a, X_b, kernel='sigmoid', iters=100):
  stat = mmd(X_a, X_b, kernel)
  pval = 0.0
  X_tot = np.concatenate((X_a, X_b), axis=0)
  for i in range(iters):
    X_a_temp, X_b_temp = train_test_split(X_tot, train_size=len(X_a)/len(X_tot))
    stat_temp = mmd(X_a_temp, X_b_temp, kernel)
    if stat <= stat_temp:
      pval += 1.0
  return pval/iters