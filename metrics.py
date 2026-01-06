from sklearn.metrics import cohen_kappa_score
import numpy as np

def calculate_metrics(preds, targets):
    """
    Computes QWK and RMSE for a set of predictions.
    """
    preds = np.array(preds)
    targets = np.array(targets)
    
    qwk = cohen_kappa_score(preds, targets, weights='quadratic', labels=np.arange(1, 10))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    
    return {"qwk": qwk, "rmse": rmse}
