import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib


# Custom Dataset class for PyTorch
class CreditCardDataset(Dataset):
    def __init__(self, X, y):
        # Convert data into tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return one sample (features, label)
        return self.X[idx], self.y[idx]


def load_data(file_path):
    # Load CSV file
    df = pd.read_csv(file_path)

    # Select important features
    features = ['V3', 'V4', 'V7', 'V10', 'V11', 'V12', 'V14', 'V16', 'Amount']
    X = df[features]
    y = df['Class']

    # Split data (before applying SMOTE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale ONLY the 'Amount' column
    scaler = StandardScaler()
    X_train['Amount'] = scaler.fit_transform(X_train[['Amount']]).astype(float)
    X_test['Amount'] = scaler.transform(X_test[['Amount']]).astype(float)
    joblib.dump(scaler, "scaler.pkl")

    # Apply SMOTE only on training data
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test


def get_datasets(file_path):
    # Get processed data
    X_train, X_test, y_train, y_test = load_data(file_path)

    # Convert to PyTorch Dataset
    train_dataset = CreditCardDataset(X_train.values, y_train.values)
    test_dataset = CreditCardDataset(X_test.values, y_test.values)

    return train_dataset, test_dataset