import streamlit as st
import torch
import numpy as np
import joblib

from model import FraudDetectionModel


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fraud Detection", layout="centered")

# ---------------- LOAD MODEL ----------------
model = FraudDetectionModel(input_dim=9)
model.load_state_dict(torch.load("fraud_model.pth"))
model.eval()

# ---------------- LOAD SCALER ----------------
scaler = joblib.load("scaler.pkl")


# ---------------- TITLE ----------------
st.title("💳 Credit Card Fraud Detection System")
st.markdown("### Detect fraudulent transactions using AI")

st.write("Fill the details below to analyze a transaction.")

# ---------------- INPUT SECTION ----------------
st.subheader("💰 Transaction Details")

Amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=0.0)


st.subheader("📊 Behavior & Risk Indicators")

V3 = st.slider("Spending Pattern Deviation", -5.0, 5.0, 0.0)
V4 = st.slider("Transaction Frequency Change", -5.0, 5.0, 0.0)
V7 = st.slider("Location/Usage Variation", -5.0, 5.0, 0.0)
V10 = st.slider("Risk Indicator 1", -5.0, 5.0, 0.0)
V11 = st.slider("Risk Indicator 2", -5.0, 5.0, 0.0)
V12 = st.slider("Risk Indicator 3", -5.0, 5.0, 0.0)
V14 = st.slider("Anomaly Score 1", -5.0, 5.0, 0.0)
V16 = st.slider("Anomaly Score 2", -5.0, 5.0, 0.0)

st.info("These values represent anonymized behavioral patterns used by the model.")


# ---------------- PREDICTION ----------------
if st.button("🔍 Check Transaction"):

    # Prepare input
    sample = np.array([[V3, V4, V7, V10, V11, V12, V14, V16, Amount]])

    # Scale only Amount
    sample[:, -1] = scaler.transform(sample[:, -1].reshape(-1, 1)).flatten()

    # Convert to tensor
    input_tensor = torch.tensor(sample, dtype=torch.float32)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    st.subheader("🧾 Result")

    if prob > 0.7:
        st.error(f"🚨 Fraudulent Transaction Detected!\n\nProbability: {prob:.4f}")
    else:
        st.success(f"✅ Legitimate Transaction\n\nProbability: {prob:.4f}")

    # Progress bar (visual confidence)
    st.progress(int(prob * 100))


# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built using PyTorch + Streamlit")