from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import ks_2samp

def evaluate(model, X, y):
    p = model.predict_proba(X)[:,1]
    roc = roc_auc_score(y, p)
    pr = precision_recall_curve(y, p)
    pr_auc = auc(pr[1], pr[0])
    ks = ks_2samp(p[y==1], p[y==0]).statistic
    return {"ROC_AUC": roc, "PR_AUC": pr_auc, "KS": ks}
