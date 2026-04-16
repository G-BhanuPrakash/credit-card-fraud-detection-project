import torch
import numpy as np
from model import FraudDetectionModel
from sklearn.preprocessing import StandardScaler


# ---------------- LOAD MODEL ----------------
model = FraudDetectionModel(input_dim=9)
model.load_state_dict(torch.load("fraud_model.pth"))
model.eval()


# ---------------- SAMPLE NEW DATA ----------------
# Example transaction (you can change values)
sample = np.array([[ 
    -2.5,   # V3
     1.2,   # V4
    -1.8,   # V7
    -2.0,   # V10
     2.1,   # V11
    -1.5,   # V12
    -2.3,   # V14
     0.5,   # V16
     1500   # Amount
]])


# ---------------- SCALE ONLY AMOUNT ----------------
scaler = StandardScaler()

# IMPORTANT: fit on sample is NOT ideal but okay for testing
sample[:, -1] = scaler.fit_transform(sample[:, -1].reshape(-1, 1)).flatten()


# ---------------- CONVERT TO TENSOR ----------------
input_tensor = torch.tensor(sample, dtype=torch.float32)


# ---------------- PREDICTION ----------------
with torch.no_grad():
    output = model(input_tensor)
    prob = torch.sigmoid(output)

    if prob.item() > 0.7:
        print("🚨 Fraud Transaction")
    else:
        print("✅ Normal Transaction")

    print(f"Fraud Probability: {prob.item():.4f}")