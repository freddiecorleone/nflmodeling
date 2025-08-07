# app.py
import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib 
import torch
from sklearn.preprocessing import StandardScaler 

from src.utils import paths
from src.NNModelCreator import WinProbNN

# Load custom CSS for font and styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("src/style.css")

# Override font globally

# Streamlit config
st.set_page_config(page_title="NFL Model", layout="centered")

# Page title
st.markdown("# 🏈 NFL Win Probability Predictor")





# Load models
model = joblib.load(paths.get_project_root() / "models" / "win_prob_model.pkl")
model2 = WinProbNN(input_dim=8)
model2.load_state_dict(torch.load("models/nn_model.pt"))
scaler = joblib.load(paths.get_project_root() /"models"/"scaler.pkl")
model2.eval()

# UI inputs


# All widgets go here
game_seconds_passed = st.slider("Seconds passed", 0, 3600, 1800)
game_seconds_remaining = 3600 - game_seconds_passed
current_margin = st.number_input("Current margin (home - away)", -50, 50, 0)
spread = st.number_input("Pregame spread (home - away)", -30.0, 30.0, 0.0)
ou = st.number_input("Over/Under", 20.0, 100.0, 45.0)
yardline_100 = st.slider("Yards to goal line", 0, 100, 50)
possessioninput = st.selectbox("Who has possession?", ["Home", "Away"])
home_possession = 1 if possessioninput == "Home" else 0
ydstogo = st.number_input("Yards to go", 1, 40, 10)
down = st.number_input("Down", 1, 4, 1)

columns = ['game_seconds_remaining', 'down', 'ydstogo', 'yardline_100',
            'home_possession', 'margin', 'total_line', 'spread_line']

row = pd.DataFrame([[
    game_seconds_remaining,
    down,
    ydstogo,
    yardline_100,
    home_possession,
    current_margin,
    ou,
    spread
]], columns=columns)

if st.button("Predict Win Probability for Home Team"):
    
    proba = model.predict_proba(row)[0][1]
    st.metric("XGBoost Win Probability", f"{proba * 100:.2f}%")

    model2.eval()
    with torch.no_grad():
        x_input = row.to_numpy()
        x_scaled = scaler.transform(x_input)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        y_proba = model2(x_tensor).item()
        st.metric("Neural Net Win Probability", f"{y_proba * 100:.2f}%")



