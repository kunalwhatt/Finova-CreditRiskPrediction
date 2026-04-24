from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def train_logistic(X, y):
    m = LogisticRegression(max_iter=1000)
    m.fit(X, y)
    return m

def train_xgb(X, y):
    m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, scale_pos_weight=5)
    m.fit(X, y)
    return m
