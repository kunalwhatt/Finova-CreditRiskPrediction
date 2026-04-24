# Credit Risk Prediction for Indian FinTech Lending Apps

End-to-end **Data Warehousing + Data Mining** project aligned with real underwriting pipelines.

## Features
- Synthetic dataset generator (25K+ rows)
- Data warehouse (Star + ER) SQL
- Feature engineering (affordability, stability, credit, device)
- Baseline (Logistic) + Advanced (XGBoost)
- Class imbalance handling (SMOTE)
- WOE/IV scorecard
- SHAP explainability (global + per-user)
- Risk banding + decision engine (Approve/Reject/Manual Review)
- Drift detection (KS test)
- Optional Streamlit dashboard

## How to Run
```bash
pip install -r requirements.txt
python generate_dataset.py
python main.py
```

## Folder Structure
- `src/` → core modules
- `sql/` → warehouse DDL
- `diagrams/` → architecture, star, ER (mermaid)
- `reports/` → documentation outline
- `data/` → generated CSV

## Author
Kunal Kumar Dappu
