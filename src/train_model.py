"""
train_model.py
Simple ML pipeline for the virtual sensor:
 - load data
 - build basic features (lag, rolling mean)
 - train a Ridge regression and a small RandomForest for comparison
 - save best model to models/
 - create basic evaluation plots in images/
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

sns.set(style="whitegrid")

DATA_PATH = Path("../data/virtual_sensor_data.csv")
OUT_MODEL = Path("../models/virtual_sensor_ridge.joblib")
OUT_IMG = Path("../images")
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

def feature_engineer(df):
    df = df.copy()
    # simple rolling features and lags
    df["accel_roll5"] = df["accel"].rolling(5, min_periods=1).mean()
    df["gyro_roll5"] = df["gyro"].rolling(5, min_periods=1).mean()
    df["strain_roll5"] = df["strain"].rolling(5, min_periods=1).mean()
    # lag features
    df["accel_lag1"] = df["accel"].shift(1).fillna(method="bfill")
    df["gyro_lag1"] = df["gyro"].shift(1).fillna(method="bfill")
    df["strain_lag1"] = df["strain"].shift(1).fillna(method="bfill")
    # optionally add ratios
    df["accel_strain_ratio"] = (df["accel"] / (df["strain"] + 1e-6)).replace([np.inf, -np.inf], 0)
    return df

def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df = feature_engineer(df)
    X = df[["accel", "gyro", "strain", "accel_roll5", "gyro_roll5", "strain_roll5",
            "accel_lag1", "gyro_lag1", "strain_lag1", "accel_strain_ratio"]].values
    y = df["hidden_load"].values
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_and_prepare()

    # Model 1: Ridge regression (fast, interpretable)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    print(f"Ridge MSE: {mse_ridge:.4f}  R2: {r2_ridge:.4f}")

    # Model 2: RandomForest (stronger baseline)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"RF MSE: {mse_rf:.4f}  R2: {r2_rf:.4f}")

    # Save the Ridge (interpretable) model
    joblib.dump(ridge, OUT_MODEL)
    print(f"Saved Ridge model to {OUT_MODEL}")

    # Plots
    t = np.arange(len(y_test))
    plt.figure(figsize=(10,4))
    plt.plot(y_test, label="Ground truth", alpha=0.8)
    plt.plot(y_pred_ridge, label="Ridge pred", alpha=0.8)
    plt.legend()
    plt.title("Ridge: Ground truth vs Prediction")
    plt.tight_layout()
    plt.savefig(OUT_IMG / "ridge_vs_truth.png")
    plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(y_test, label="Ground truth", alpha=0.8)
    plt.plot(y_pred_rf, label="RF pred", alpha=0.8)
    plt.legend()
    plt.title("RF: Ground truth vs Prediction")
    plt.tight_layout()
    plt.savefig(OUT_IMG / "rf_vs_truth.png")
    plt.close()

if __name__ == "__main__":
    train_and_evaluate()