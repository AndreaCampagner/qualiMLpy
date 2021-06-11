from sklearn.metrics import brier_score_loss
import numpy as np

def var_brier_score(y_test, y_proba):
  val = np.mean(y_proba**4)
  val -= 4*np.mean(y_proba**3 * y_test)
  val += 6*np.mean(y_proba**2 * y_test)
  val -= 4*np.mean(y_proba * y_test)
  val += np.mean(y_test) 
  val -= brier_score_loss(y_test, y_proba)**2
  return val

def var_auc(auc, num_pos, num_neg):
  prop = num_pos/(num_pos + num_neg)
  n = num_pos + num_neg
  se = np.sqrt(auc*(1-auc)*(1 + (n/2 - 1)*(1 - auc)/(2- auc) + (n/2 -1)*auc/(1+auc))/(n**2*prop*(1-prop)))
  return se**2

def var_nb(sens, spec, prop, n, th=0.5):
  w = (1-prop)/prop*th/(1-th)
  return 1/n * ( sens*(1-sens)/prop + w**2 * spec*(1-spec)/(1-prop) + w**2 * (1-spec)**2/(prop*(1-prop)))
