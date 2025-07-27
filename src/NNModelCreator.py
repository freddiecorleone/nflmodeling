import torch, joblib, sqlite3, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from utils import paths 


class WinProbNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)  # one output for binary classification

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))  # output is a probability
        return x



def createNNModel():

   

    modelingpath = paths.get_project_root() / "data" / "modeling.db"
    conn = sqlite3.connect(modelingpath)
    df = pd.read_sql("SELECT * FROM plays_for_model", conn)
    conn.close()
    X = df.drop(columns=["result", "game_id"])  # exclude ID and non-numeric columns
    y = df["result"]
    print(X.head(20))
    print(y.head(20))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_array = y.values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_array, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    print('working?')

    model = WinProbNN(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()  # binary cross-entropy

    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_tensor)
        loss = loss_fn(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_tensor)
                val_loss = loss_fn(val_pred, y_val_tensor)
            print(f"Epoch {epoch}: train loss = {loss.item():.4f}, val loss = {val_loss.item():.4f}")
        

    model.eval()
    with torch.no_grad():
        y_val_proba = model(X_val_tensor).numpy().flatten()

    roc_auc = roc_auc_score(y_val, y_val_proba)
    print(f"ROC AUC: {roc_auc:.4f}")
    torch.save(model.state_dict(), paths.get_project_root() / "models/nn_model.pt")
    joblib.dump(scaler, paths.get_project_root() / "models/scaler.pkl")




