# =========================
# SAFE MATPLOTLIB BACKEND
# =========================
import matplotlib
matplotlib.use("Agg")

# =========================
# IMPORTS
# =========================
from flask import Flask, render_template
import matplotlib.pyplot as plt
import os
import torch
import joblib
import numpy as np
import pandas as pd
from torch import nn

# =========================
# GLOBAL SETTINGS
# =========================
WINDOW_SIZE = 20
current_index = 0   # pointer for sliding window

CSV_PATH = "EV_Battery_Charging_TR_Dataset_with_Notes.csv"

# =========================
# MODEL DEFINITION
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze()


# =========================
# LOAD MODEL & SCALERS
# =========================
model = LSTMModel(input_size=11)
model.load_state_dict(torch.load("lstm_ev_thermal_model.pt", map_location="cpu"))
model.eval()

feature_scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")


# =========================
# LOAD DATA + SLIDING WINDOW
# =========================
def load_sliding_window(csv_path):
    global current_index

    df = pd.read_csv(csv_path)

    # Sort by time if timestamp exists
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp")
        df["delta_time"] = df["Timestamp"].diff().dt.total_seconds().fillna(1)
    else:
        df["delta_time"] = 1

    # Feature engineering
    df["temp_rise_rate"] = df["AvgTemp_C"].diff() / df["delta_time"]
    df["temp_rise_rate"] = df["temp_rise_rate"].fillna(0)

    df["current_stress"] = df["ChargeCurrent_A"] / (df["SOC_%"] + 1)

    df["voltage_mean"] = df["PackVoltage_V"].rolling(20, min_periods=1).mean()
    df["voltage_deviation"] = df["PackVoltage_V"] - df["voltage_mean"]

    df["temp_spread"] = df["MaxTemp_C"] - df["AvgTemp_C"]

    feature_cols = [
        "SOC_%",
        "ChargeCurrent_A",
        "PackVoltage_V",
        "AvgTemp_C",
        "MaxTemp_C",
        "InternalResistance_mOhm",
        "ChargePower_kW",
        "temp_rise_rate",
        "current_stress",
        "voltage_deviation",
        "temp_spread"
    ]

    df = df[feature_cols].dropna().reset_index(drop=True)

    if len(df) < WINDOW_SIZE:
        raise ValueError("Not enough rows in CSV for sliding window")

    # Wrap around when reaching end
    if current_index + WINDOW_SIZE >= len(df):
        current_index = 0

    window_df = df.iloc[current_index : current_index + WINDOW_SIZE]
    history_df = df.iloc[: current_index + WINDOW_SIZE].tail(50)

    current_index += 1

    window = window_df.values.reshape(1, WINDOW_SIZE, len(feature_cols))
    return window, history_df


# =========================
# PLOT GENERATION
# =========================
def generate_plots(actual_series, predicted_value):
    os.makedirs("static", exist_ok=True)

    predicted_series = np.full_like(actual_series, predicted_value)
    residual_series = actual_series - predicted_value

    # Temperature plot
    plt.figure(figsize=(6, 3))
    plt.plot(actual_series, label="Actual Temp")
    plt.plot(predicted_series, label="Predicted Temp")
    plt.legend()
    plt.title("Temperature Trend")
    plt.tight_layout()
    plt.savefig("static/temp_plot.png")
    plt.close()

    # Residual plot
    plt.figure(figsize=(6, 3))
    plt.plot(residual_series, color="red", label="Residual")
    plt.axhline(0, linestyle="--", color="black")
    plt.legend()
    plt.title("Residual Trend")
    plt.tight_layout()
    plt.savefig("static/residual_plot.png")
    plt.close()


# =========================
# FLASK APP
# =========================
app = Flask(__name__)

@app.route("/")
def home():
    try:
        # Load sliding window
        input_window, history = load_sliding_window(CSV_PATH)

        # Scale features
        reshaped = input_window.reshape(-1, 11)
        scaled = feature_scaler.transform(reshaped)
        scaled = scaled.reshape(1, WINDOW_SIZE, 11)

        x = torch.tensor(scaled, dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = model(x).numpy()

        predicted_temp = target_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)
        )[0][0]

        actual_temp = input_window[0, -1, 3]
        residual = actual_temp - predicted_temp

        # Generate plots
        actual_series = history["AvgTemp_C"].values
        generate_plots(actual_series, predicted_temp)

        # Risk classification
        if residual > 10:
            risk = "ALARM"
        elif residual > 6:
            risk = "WARNING"
        elif residual > 3:
            risk = "WATCH"
        else:
            risk = "NORMAL"

        return render_template(
            "index.html",
            predicted=round(predicted_temp, 2),
            actual=round(actual_temp, 2),
            residual=round(residual, 2),
            risk=risk
        )

    except Exception as e:
        return f"<pre>{str(e)}</pre>"


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
