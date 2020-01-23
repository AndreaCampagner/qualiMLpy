
def representativeness(dataset, X_train, X_test)
	"""Computes the degree to which X_test is representative of X_train, by comparing their distribution of distances"""
	distrib_train = pdist(X_train).flatten()

	train_idx, test_idx = linear_sum_assignment(cdist(X_train, X_test))
	X_new_train = X_train.copy()
	X_new_train[train_idx,:] = X_test[test_idx,:]
	distrib_new_train = pdist(X_new_train).flatten()

	best = np.max(np.abs(distrib_new_train - distrib_train))

	iters = 100
	count = 0
	for r in range(iters):
		print(r)
		train, test = train_test_split(dataset, test_size=0.33, random_state = r)
		distrib_train_temp = pdist(train).flatten()
		train_idx, test_idx = linear_sum_assignment(cdist(train, test))
		X_new_train = train.copy()
		X_new_train[train_idx,:] = test[test_idx,:]
		distrib_new_train_temp = pdist(X_new_train).flatten()
		
		if best <= np.max(np.abs(distrib_new_train_temp - distrib_train_temp)):
			count += 1
	return count/iters