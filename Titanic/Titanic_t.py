import os
from random import shuffle
from matplotlib.pylab import astype
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

train_df["Age"].fillna(train_df["Age"].mean(), inplace=True)
test_df["Age"].fillna(test_df["Age"].mean(), inplace=True)

train_df["Embarked"].fillna(train_df["Embarked"].mode(), inplace=True)
test_df["Embarked"].fillna(test_df["Embarked"].mode(), inplace=True)

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

encoder = LabelEncoder()
train_df["Sex"] = encoder.fit_transform(train_df["Sex"])
test_df["Sex"] =  encoder.transform(test_df["Sex"])

train_df = pd.get_dummies(train_df, columns=["Embarked"], drop_first=True)
test_df = pd.get_dummies(test_df, columns=["Embarked"], drop_first=True)

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S"]
X = train_df[features]
y = train_df["Survived"]

scaler = StandardScaler()
X = scaler.fit_transform(X)
test_X = scaler.transform(test_df[features])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

train_dataset = TitanicDataset(X_train, y_train)
val_dataset = TitanicDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)
    
def train(model, train_loader, val_loader, criterion, optimizer, epochs=10, device="cpu"):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            predicted = (y_pred > 0.5).float()  # 확률값을 0 또는 1로 변환
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        
        train_accuracy = correct / total
            
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")


def evaluate(model, val_loader, criterion, device="cpu"):
    model.eval()
    total_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            
            predicted = (y_pred > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    val_accuracy = correct / total  # 검증 정확도
    return total_loss / len(val_loader), val_accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TitanicModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
train(model, train_loader, val_loader, criterion, optimizer, epochs=500, device=device)

model.eval()
test_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
with torch.no_grad():
    test_preds = model(test_tensor).cpu().numpy()
test_preds = (test_preds > 0.5).astype(int).flatten()

submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": test_preds})
submission.to_csv("submission_torch.csv", index=False)


print("First Layer Weights:", model.model[0].weight.data)
print("First Layer Bias:", model.model[0].bias.data)

for name, param in model.state_dict().items():
    print(f"{name}: {param.shape}")

model_path = "titanic_model_torch_4.pth"
if not os.path.exists(model_path):
    with open(model_path, "wb") as f:
        torch.save(model.state_dict(), f)
    print(f"Model saved as {model_path}")
else:
    print(f"{model_path} already exists.")
    
submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": test_preds})
submission.to_csv("submission2.csv", index=False)