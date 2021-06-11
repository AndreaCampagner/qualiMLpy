
import numpy as np

def mss_auc(auc, prop, target_se = 0.0255):
  for n in range(10, 1000000):
    se = np.sqrt(auc*(1-auc)*(1 + (n/2 - 1)*(1 - auc)/(2- auc) + (n/2 -1)*auc/(1+auc))/(n**2*prop*(1-prop)))
    if se <= target_se:
      return n

def mss_nb(sens, spec, prop, th=0.5, target_se = 0.051):
  w = (1-prop)/prop*th/(1-th)
  return 1/target_se**2 * ( sens*(1-sens)/prop + w**2 * spec*(1-spec)/(1-prop) + w**2 * (1-spec)**2/(prop*(1-prop)))

def mss_brier(var_briers, target_conf):
  return (2*1.96*np.sqrt(var_briers)/target_conf)**2