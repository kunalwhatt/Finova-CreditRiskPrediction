import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 25000

age = np.random.randint(21, 60, N)
income = np.clip(np.random.normal(40000, 15000, N), 10000, 150000)
employment_type = np.random.choice(['salaried','self-employed','student'], N, p=[0.6,0.3,0.1])
loan_amount = np.random.randint(5000, 100000, N)
tenure_months = np.random.choice([3,6,9,12], N)
emi = loan_amount/tenure_months + np.random.normal(500,200,N)

credit_history_months = np.random.randint(1,60,N)
active_loans = np.random.randint(0,5,N)
past_delinquency = np.random.choice([0,1],N,p=[0.75,0.25])

new_device_flag = np.random.choice([0,1],N,p=[0.8,0.2])
kyc_mismatch = np.random.choice([0,1],N,p=[0.9,0.1])
recent_txn_velocity = np.random.randint(1,50,N)

emi_income_ratio = emi/income
loan_income_ratio = loan_amount/income

risk_score = (
    0.35*emi_income_ratio +
    0.25*loan_income_ratio +
    0.15*past_delinquency +
    0.10*new_device_flag +
    0.10*kyc_mismatch +
    0.05*(1/(credit_history_months+1))
)

risk_score += np.random.normal(0,0.05,N)
prob_default = 1/(1+np.exp(-5*(risk_score-0.5)))
default = (prob_default > 0.5).astype(int)

df = pd.DataFrame({
    "age": age,
    "income": income.astype(int),
    "employment_type": employment_type,
    "loan_amount": loan_amount,
    "tenure_months": tenure_months,
    "emi": emi.astype(int),
    "credit_history_months": credit_history_months,
    "active_loans": active_loans,
    "past_delinquency": past_delinquency,
    "new_device_flag": new_device_flag,
    "kyc_mismatch": kyc_mismatch,
    "recent_txn_velocity": recent_txn_velocity,
    "emi_income_ratio": emi_income_ratio,
    "loan_income_ratio": loan_income_ratio,
    "default": default
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/credit_risk_dataset_25k.csv", index=False)
print("Dataset generated at data/credit_risk_dataset_25k.csv")
print(df.head())
