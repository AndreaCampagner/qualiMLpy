from sklearn.metrics import recall_score
import numpy as np

def nb(y_true, y_proba, th=0.5):
  prop = len(y_true[y_true == 1])/len(y_true)
  y_pred = (y_proba >= th).astype(int)
  sens = recall_score(y_true, y_pred)
  spec = recall_score(y_true, y_pred, pos_label=0)
  return (sens*prop - (1-spec)*(1-prop)*th/(1-th))/prop

def wu(y_true, y_proba, ths=None, relevances=None):
  if ths is None and relevances is None:
    return nb(y_true, y_proba)
  
  if np.isscalar(ths) and relevances is None:
    return nb(y_true, y_proba, ths)
  
  if relevances is None:
    relevances = np.ones(y_proba.shape)
  
  if ths is None:
    ths = np.ones(y_proba.shape)*0.5
  elif np.isscalar(ths):
    ths = np.ones(y_proba.shape)*ths


  if len(ths) != len(y_true):
    raise ValueError("If not scalar or None, ths should have the same length as y_true")
  if len(relevances) != len(y_true):
    raise ValueError("If not None, relevances should have the same length as y_true")


  pos_idx = y_true == 1
  rs = np.sum(relevances[pos_idx])
  pp = y_proba >= ths
  tp = np.logical_and(pos_idx, pp)
  fp = np.logical_and(np.logical_not(pos_idx), pp)
  return np.sum(tp*relevances)/rs - np.sum(ths/(1-ths)*fp*relevances)/rs
  
