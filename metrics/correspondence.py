import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def degree_correspondance(X_a, X_b, metric='euclidean', iters = 100, averaged=False):
	similarities = []
	if not averaged:
		return psi_similarity_test(X_a, X_b, metric, iters)
	else:
		for ind in range(len(X_b)):
			similarities.append(psi_similarity_test(X_a, [X_b[ind]], metric, iters))
		return np.mean(similarities)

def psi_similarity_test(X_a, X_b, metric='euclidean', iters=100):
  stat = psi_similarity_helper(X_a, X_b, metric)
  pval = 0.0
  X_tot = np.concatenate((X_a, X_b), axis=0)
  for i in range(iters):
    X_a_temp, X_b_temp = train_test_split(X_tot, train_size=len(X_a)/len(X_tot))
    stat_temp = psi_similarity_helper(X_a_temp, X_b_temp, metric)
    if stat <= stat_temp:
      pval += 1.0
  return pval/iters

def psi_similarity_helper(X_a, X_b, metric='euclidean'):
  #intra = np.sum(pairwise_distances(X_a, X_a, metric=metric))
  nn = NearestNeighbors()
  nn.fit(X_a)

  stat = 0
  for x in X_b:
    idx = nn.kneighbors([x], 1, return_distance=False)
    temp = 0
    temp -= np.sum(pairwise_distances(X_a, X_a[idx][0], metric=metric))
    temp += np.sum(pairwise_distances(X_a, [x], metric=metric))
    stat += temp
  stat /= len(X_b)
  return stat

