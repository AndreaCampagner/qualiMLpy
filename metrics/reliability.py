import numpy as np

def irr(preds, confs = None, accs = None, cohers=None):
	res_min = 0
	res_max = 0
	if confs is None:
		confs = np.ones(preds.shape)*0.5
	if cohers is None:
		cohers = np.zeros(preds.shape[1])
	if accs is None:
		accs = np.ones(preds.shape[1])
	for row in range(len(preds)):
		count_min = 0
		count_max = 0
		r = preds[row]
		den = 0
		for i in range(len(r)):
			for j in range(len(r)):
				if i != j:
					den += 1
					if r[i] == r[j]:
						count_min_temp = count_sigma(confs[row, i], confs[row, j],
												cohers[i], cohers[j])
						count_max_temp = count_sigma(confs[row, i], confs[row, j],
												cohers[i], cohers[j], mode='max')
						acc = accs[i]*accs[j]
						acc /= (accs[i]*accs[j] + (1-accs[i])*(1-accs[j]))
						count_min_temp *= acc
						count_max_temp *= acc
						count_min += count_min_temp
						count_max += count_max_temp
						#count += countn
		count_min /= den
		count_max /= den
		res_min += count_min
		res_max += count_max
	return res_min/preds.shape[0], res_max/preds.shape[0]
	
def count_sigma(conf_i, conf_j, cohers_i, cohers_j, mode='min'):
	count = 0
	conf_corr_i = conf_i
	conf_corr_j = conf_j
	if mode == 'min':
		conf_corr_i -= cohers_i
		conf_corr_j -= cohers_j
	else:
		conf_corr_i += cohers_i
		conf_corr_j += cohers_j
		
	if conf_corr_i < 0:
		conf_corr_i = 0
	if conf_corr_j < 0:
		conf_corr_j = 0
	if conf_corr_i > 1:
		conf_corr_i = 1
	if conf_corr_j > 1:
		conf_corr_j = 1		
		
	countn = conf_corr_i*conf_corr_j
	countd = conf_corr_i*conf_corr_j
	countd += conf_corr_i*(1 - conf_corr_j)/2
	countd += conf_corr_j*(1 - conf_corr_i)/2
	countd += (1- conf_corr_i)*(1 - conf_corr_j)/4
	count += countn/countd
	return count