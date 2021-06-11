from sklearn.metrics import recall_score

def nb(y_true, y_proba, th=0.5):
  prop = len(y_true[y_true == 1])/len(y_true)
  y_pred = (y_proba >= th).astype(int)
  sens = recall_score(y_true, y_pred)
  spec = recall_score(y_true, y_pred, pos_label=0)
  return (sens*prop - (1-spec)*(1-prop)*th/(1-th))/prop