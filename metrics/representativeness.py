from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp
from sklearn.utils import resample
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.metrics.pairwise import pairwise_kernels

def bootstrap(stat_value, data, statistic, iters=100):
    pvalue = 0
    for i in range(iters):
        train, test = train_test_split(data, test_size=0.33, random_state = i)
        stat = statistic(train,test)
        if  stat >= stat_value:
            pvalue += 1
    return pvalue/100

def mmd(X_train, X_test, kernel='sigmoid'):
    intra = np.sum(pairwise_kernels(X_train, X_train, metric=kernel))
    intra += np.sum(pairwise_kernels(X_test, X_test, metric=kernel))
    extra = np.sum(pairwise_kernels(X_train, X_test, metric=kernel))
    return intra - 2*extra

def phi(X_train, X_test, statistic, reshape=False):
    """Computes the degree to which X_test is representative of X_train, by comparing their distribution of distances"""
    distrib_train = pdist(X_train).flatten()

    train_idx, test_idx = linear_sum_assignment(cdist(X_train, X_test))
    X_new_train = X_train.copy()
    X_new_train[train_idx,:] = X_test[test_idx,:]
    distrib_new_train = pdist(X_new_train).flatten()
    stat = 0
    if reshape == True:
        stat = statistic(distrib_train.reshape(1,-1)[0], distrib_new_train.reshape(1,-1)[0])
    else:
        stat = statistic(distrib_train.reshape(1,-1), distrib_new_train.reshape(1,-1))
    #print(stat)
    return stat

def ks_2samp_supp(distrib_1, distrib_2):
    return ks_2samp(distrib_1, distrib_2).statistic

def phi_ks(X_train, X_test):
    return phi(X_train, X_test, ks_2samp_supp, reshape=True)

def phi_js(X_train, X_test):
    return phi(X_train, X_test, jensenshannon, reshape=True)

def phi_mmd(X_train, X_test):
    return phi(X_train, X_test, mmd, reshape=False)
