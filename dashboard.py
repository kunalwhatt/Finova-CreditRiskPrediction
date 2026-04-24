import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from datetime import date
from io import BytesIO
import tempfile

# ---------------- SAFE IMPORTS ----------------
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except:
    REPORTLAB_AVAILABLE = False

from src.data_preprocessing import basic_cleaning, encode_categorical
from src.feature_engineering import create_features
from src.model_training import train_xgb
from src.risk_banding import assign_risk_band, decision_engine

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Finova", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
body {background-color: #f8fafc;}

.card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    text-align:center;
}

.green {color:#16a34a;font-weight:700;}
.yellow {color:#ca8a04;font-weight:700;}
.red {color:#dc2626;font-weight:700;}

.reason-box {
    background: #ffffff;
    padding:15px;
    border-radius:12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("Finova")
st.caption("AI-Driven Credit Risk Engine")

# ---------------- LOAD ----------------
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

# ---------------- INPUT ----------------
st.subheader("Applicant Details")

col1, col2 = st.columns(2)
name = col1.text_input("Full Name")
dob = col2.date_input("Date of Birth", date(2000,1,1))

# Auto age
today = date.today()
age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
st.write(f"Age: {age}")

col3, col4, col5 = st.columns(3)
income = col3.number_input("Income", 10000, 150000, 40000)
loan_amount = col4.number_input("Loan Amount", 5000, 100000, 20000)
emi = col5.number_input("EMI", 1000, 50000, 5000)

employment = st.selectbox("Employment", ["salaried","self-employed","student"])
credit_history = st.slider("Credit History", 1, 60, 12)
active_loans = st.slider("Active Loans", 0, 5, 1)
delinquency = st.selectbox("Past Delinquency", [0,1])

# Prepare input
input_df = pd.DataFrame([{
    "age": age,
    "income": income,
    "employment_type": employment,
    "loan_amount": loan_amount,
    "tenure_months": 12,
    "emi": emi,
    "credit_history_months": credit_history,
    "active_loans": active_loans,
    "past_delinquency": delinquency,
    "new_device_flag": 0,
    "kyc_mismatch": 0,
    "recent_txn_velocity": 10
}])

input_df = create_features(input_df)
input_encoded = encode_categorical(input_df)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# ---------------- RUN ----------------
if st.button("Run Risk Assessment"):

    prob = model.predict_proba(input_encoded)[0][1]
    risk = assign_risk_band(prob)
    decision = decision_engine(risk)

    color = "green" if risk=="LOW" else "yellow" if risk=="MEDIUM" else "red"

    # ---------- METRICS ----------
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='card'><h4>PD</h4><h2>{prob:.2f}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><h4>Risk</h4><h2 class='{color}'>{risk}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'><h4>Decision</h4><h2 class='{color}'>{decision}</h2></div>", unsafe_allow_html=True)

    # ---------- GAUGE ----------
    st.subheader("Risk Score")
    st.progress(float(prob))

    # ---------- SHAP ----------
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(input_encoded)
        shap_ok = True
    except:
        shap_ok = False

    st.subheader("Key Reasons")

    if shap_ok:
        vals = shap_values[0].values
        features = input_encoded.columns

        importance = sorted(
            zip(features, vals),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
    else:
        importance = []

    reasons = []
    for f, v in importance:
        reason = f"{f.replace('_',' ').title()} influenced the risk ({'increase' if v>0 else 'decrease'})"
        reasons.append(reason)
        st.markdown(f"<div class='reason-box'>{reason}</div>", unsafe_allow_html=True)

    if not reasons:
        st.info("Basic rule-based insights applied.")

    # ---------- CHART ----------
    if shap_ok:
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp.name)

    # ---------- CSV ----------
    result_df = pd.DataFrame([{
        "Name": name,
        "PD": prob,
        "Risk": risk,
        "Decision": decision
    }])

    st.download_button("Download CSV", result_df.to_csv(index=False), "result.csv")

    # ---------- PDF ----------
    if REPORTLAB_AVAILABLE:

        def create_pdf():
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()

            content = [
                Paragraph("Finova Loan Report", styles['Title']),
                Spacer(1,12),
                Paragraph(f"Name: {name}", styles['Normal']),
                Paragraph(f"PD: {prob:.2f}", styles['Normal']),
                Paragraph(f"Risk: {risk}", styles['Normal']),
                Paragraph(f"Decision: {decision}", styles['Normal']),
                Spacer(1,12),
                Paragraph("Key Reasons:", styles['Heading2'])
            ]

            for r in reasons:
                content.append(Paragraph(r, styles['Normal']))

            if shap_ok:
                content.append(Spacer(1,12))
                content.append(Image(tmp.name, width=400, height=200))

            doc.build(content)
            buffer.seek(0)
            return buffer

        pdf = create_pdf()
        st.download_button("Download PDF Report", pdf, "loan_report.pdf")

    else:
        st.warning("PDF export not available (install reportlab to enable)")