import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from src.data_preprocessing import basic_cleaning, encode_categorical
from src.feature_engineering import create_features
from src.model_training import train_xgb
from src.risk_banding import assign_risk_band, decision_engine

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("Credit Risk Prediction Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("data/credit_risk_dataset_25k.csv")
    df = basic_cleaning(df)
    df = create_features(df)
    return df

@st.cache_resource
def train_model(df):
    df_encoded = encode_categorical(df)
    X = df_encoded.drop("default", axis=1)
    y = df_encoded["default"]
    model = train_xgb(X, y)
    return model, X.columns

df = load_data()
model, feature_columns = train_model(df)

st.sidebar.header("Applicant Details")

age = st.sidebar.slider("Age", 21, 60, 30)
income = st.sidebar.number_input("Income", 10000, 150000, 40000)
loan_amount = st.sidebar.number_input("Loan Amount", 5000, 100000, 20000)
tenure = st.sidebar.selectbox("Tenure", [3,6,9,12])
emi = st.sidebar.number_input("EMI", 1000, 50000, 5000)

employment = st.sidebar.selectbox("Employment", ["salaried","self-employed","student"])

credit_history = st.sidebar.slider("Credit History", 1, 60, 12)
active_loans = st.sidebar.slider("Active Loans", 0, 5, 1)
past_delinquency = st.sidebar.selectbox("Past Delinquency", [0,1])

new_device = st.sidebar.selectbox("New Device", [0,1])
kyc = st.sidebar.selectbox("KYC Mismatch", [0,1])
txn_velocity = st.sidebar.slider("Txn Velocity", 1, 50, 10)

input_df = pd.DataFrame([{
    "age": age,
    "income": income,
    "employment_type": employment,
    "loan_amount": loan_amount,
    "tenure_months": tenure,
    "emi": emi,
    "credit_history_months": credit_history,
    "active_loans": active_loans,
    "past_delinquency": past_delinquency,
    "new_device_flag": new_device,
    "kyc_mismatch": kyc,
    "recent_txn_velocity": txn_velocity
}])

input_df = create_features(input_df)
input_encoded = encode_categorical(input_df)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

if st.button("Predict Risk"):
    prob = model.predict_proba(input_encoded)[0][1]
    risk = assign_risk_band(prob)
    decision = decision_engine(risk)

    st.write(f"Probability of Default: {prob:.2f}")
    st.write(f"Risk Band: {risk}")
    st.write(f"Decision: {decision}")

    st.subheader("Explainability")
    explainer = shap.Explainer(model)
    shap_values = explainer(input_encoded)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Default Distribution")
st.bar_chart(df['default'].value_counts())
