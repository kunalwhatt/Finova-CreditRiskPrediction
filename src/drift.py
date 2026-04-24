from scipy.stats import ks_2samp

def detect_drift(train_scores, new_scores):
    stat, p = ks_2samp(train_scores, new_scores)
    return {"ks": stat, "p_value": p, "drift": p < 0.05}
