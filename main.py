import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preprocessing import basic_cleaning, encode_categorical
from src.feature_engineering import create_features, select_features
from src.imbalance_handling import apply_smote
from src.model_training import train_logistic, train_xgb
from src.evaluation import evaluate
from src.risk_banding import assign_risk_band, decision_engine
from src.explainability import compute_shap, save_summary_plot

DATA_PATH = "data/credit_risk_dataset_25k.csv"

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Run generate_dataset.py first.")

    df = pd.read_csv(DATA_PATH)
    df = basic_cleaning(df)
    df = create_features(df)
    df = encode_categorical(df)

    X, y = select_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    log = train_logistic(X_train_bal, y_train_bal)
    xgb = train_xgb(X_train_bal, y_train_bal)

    print("Logistic:", evaluate(log, X_test, y_test))
    print("XGBoost:", evaluate(xgb, X_test, y_test))

    # Risk banding
    probs = xgb.predict_proba(X_test)[:,1]
    results = pd.DataFrame({
        "PD": probs
    })
    results["Risk"] = results["PD"].apply(assign_risk_band)
    results["Decision"] = results["Risk"].apply(decision_engine)

    os.makedirs("reports", exist_ok=True)
    results.to_csv("reports/predictions.csv", index=False)

    # SHAP (sample for speed)
    sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    shap_values = compute_shap(xgb, sample)
    save_summary_plot(shap_values, sample, path="reports/shap_summary.png")

    print("Saved reports/predictions.csv and reports/shap_summary.png")

if __name__ == "__main__":
    main()
