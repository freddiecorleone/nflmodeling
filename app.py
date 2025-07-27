# app.py
import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
from src.utils import paths
import joblib 
import torch, numpy as np
from sklearn.preprocessing import StandardScaler 
from src.NNModelCreator import WinProbNN

# Load your model

model = joblib.load(paths.get_project_root() / "models" / "win_prob_model.pkl")  # or use joblib.load if saved that way
model2 = WinProbNN(input_dim=8)
model2.load_state_dict(torch.load("models/nn_model.pt"))
scaler = joblib.load(paths.get_project_root() /"models"/"scaler.pkl")
model2.eval()

# UI inputs
st.title("NFL Win Probability Predictor")

modeltype = st.selectbox("Model type", ['Neural Net', 'XGBoost'])



game_seconds_passed = st.slider("Seconds passed", 0, 3600, 1800)
game_seconds_remaining = 3600 - game_seconds_passed

current_margin = st.number_input("Current margin (home - away)", -50, 50, 0)
spread = st.number_input("Pregame spread (home - away)", -30.0, 30.0, 0.0)
ou = st.number_input("Over/Under", 20.0, 100.0, 45.0)
yardline_100 = st.slider("Field Position (0-100)", 0, 100, 50)
possessioninput = st.selectbox("Who has possession?", ["Home", "Away"])

if possessioninput == "Home":
    home_possession = 1
else:
    home_possession = 0 
ydstogo = st.number_input("Yards to go", 1, 40, 10)
down = st.number_input("Down", 1, 4, 1)
columns = ['game_seconds_remaining', 'down', 'ydstogo', 'yardline_100','home_possession', 'margin', 'total_line', 'spread_line']
    
# Feature formatting
row = pd.DataFrame([[
    game_seconds_remaining,   #seconds remaining
    down,     #fdown
    ydstogo,     # to go
    yardline_100,     # 
    home_possession,   # field_position relative to team with possession
    current_margin,     #margin home -away
    ou,  #pre game over under
    spread   # pregame spread

]], columns=columns)



# Prediction
if st.button("Predict Win Probability"):
    if modeltype == 'XGBoost':
        proba = model.predict_proba(row)[0][1]  # P(home wins)
        st.metric("XGBoost Probability", f"{proba * 100:.2f}%")
    else:
        model2.eval()
        with torch.no_grad():
            x_input = row.to_numpy()
            print(x_input)
            x_scaled = scaler.transform(x_input)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32) 
            y_proba = model2(x_tensor).item()  # returns a single float
            st.metric("Neural Net win probability",  f"{y_proba*100:.2f}%")

