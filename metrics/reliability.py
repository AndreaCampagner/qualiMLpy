import numpy as np

def irr(preds, confs = None, cohers = None, accs = None):
    res = 0
    if confs is None:
        confs = np.ones(preds.shape)*0.5
    if cohers is None:
        cohers = np.ones(preds.shape[1])
    if accs is None:
        accs = np.ones(preds.shape[1])
    for row in range(len(preds)):
        count = 0
        r = preds[row]
        den = 0
        for i in range(len(r)):
            for j in range(len(r)):
                if i != j:
                    den += 1
                    if r[i] == r[j]:
                        countn = confs[row,i]*confs[row,j]
                        countd = confs[row,i]*confs[row,j]
                        #countd = 0
                        countd += confs[row,i]*(1 - confs[row,j])/2
                        countd += confs[row,j]*(1 - confs[row,i])/2
                        countd += (1- confs[row,i])*(1 - confs[row,j])/4
                        count += countn/countd
						count *= cohers[i]*cohers[j]*accs[i]*accs[j]
                        #count += countn
        count /= den
        res += count
    return res/preds.shape[0]