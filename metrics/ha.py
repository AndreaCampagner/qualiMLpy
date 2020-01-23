import numpy as np

def ha(y_true, y_pred, t = 0.5, weights = None, diffs = None):
    real = np.array(y_true)
    res = 0
    classes = np.unique(real)
    
    if weights is None:
        weights = [1/len(classes) for c in classes]
    w = np.array(weights)
    
    if diffs is None:
        diffs = [1 for y in y_true]
    d = np.array(diffs)  
    
    preds = np.array(y_pred)
    
    for c in classes:
        idx = np.argwhere(real == c).flatten()
        c_spec = np.sum(sigma_2(real[idx], preds[idx,:], t)*d[idx])/np.sum(d[idx])
        res += c_spec*w[c]
    return res


def sigma(real, preds, thresh):
    def sigma_helper(real, pred, thresh):
        idx = np.argmax(pred)
        if real != idx:
            return 0
        if real == idx and pred[idx] > thresh:
            return 1
        res = (pred[idx] - 1/len(pred))/(thresh - 1/len(pred))
        res = 1 if res > 1 else (0 if res < 0 else res)
        return res
    
    return [sigma_helper(real[i], preds[i], thresh) for i in range(len(real))]