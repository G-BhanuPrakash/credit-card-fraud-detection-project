import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import get_datasets
from model import FraudDetectionModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

def train():
    # Path to dataset
    file_path = "data/creditcard.csv"

    # Load datasets
    train_dataset, test_dataset = get_datasets(file_path)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = FraudDetectionModel(input_dim=9).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20

    # ================= TRAINING =================
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_batch)

            # Loss calculation
            loss = criterion(outputs, y_batch)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # ================= EVALUATION =================
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []   # ✅ define OUTSIDE loop

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)
            probs = torch.sigmoid(outputs)

            preds = (probs > 0.7).float()

            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())   # append here

    print("\nResults:")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))
    print("ROC-AUC :", roc_auc_score(y_true, y_probs))  # use probs

    # Save model
    torch.save(model.state_dict(), "fraud_model.pth")
    print("\nModel saved as fraud_model.pth")


if __name__ == "__main__":
    train()