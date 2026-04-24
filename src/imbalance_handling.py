from imblearn.over_sampling import SMOTE

def apply_smote(X, y):
    sm = SMOTE()
    return sm.fit_resample(X, y)
