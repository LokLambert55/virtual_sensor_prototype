"""
generate_data.py
Generates a synthetic dataset for a virtual sensor prototype.
The 'hidden' target variable is a physical load (e.g. torque) that is not
measured directly. We simulate several indirect sensors (accel, gyro, strain)
and add noise / drift to mimic real-world data.
Outputs a CSV in ../data/virtual_sensor_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_dataset(n_samples=10000, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, 100, n_samples)  # time axis in seconds

    # True hidden variable: e.g. load signal with slow trend + bursts
    trend = 0.2 * np.sin(0.02 * t)            # slow background
    bursts = 2.0 * (np.sin(2*np.pi*0.5*t) > 0.95).astype(float)  # rare bursts
    stochastic = 0.3 * np.random.randn(n_samples)
    hidden_load = 5.0 + trend + bursts + stochastic

    # Indirect sensors: mixtures + delays + noise
    accel = 0.5 * hidden_load + 0.3 * np.sin(2*np.pi*1.0*t) + 0.2*np.random.randn(n_samples)
    gyro = 0.2 * hidden_load + 0.1 * np.sin(2*np.pi*0.3*t + 0.5) + 0.15*np.random.randn(n_samples)
    strain = 0.8 * hidden_load + 0.05 * np.cos(2*np.pi*0.2*t) + 0.25*np.random.randn(n_samples)

    # Simulate sensor drift and offset on one channel
    drift = 0.001 * t  # slow linear drift
    accel_drift = accel + drift

    # package into DataFrame
    df = pd.DataFrame({
        "time": t,
        "hidden_load": hidden_load,
        "accel": accel_drift,
        "gyro": gyro,
        "strain": strain,
    })

    return df

def save_dataset(path_out="../data/virtual_sensor_data.csv", n_samples=10000):
    out = Path(path_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(n_samples=n_samples)
    df.to_csv(out, index=False)
    print(f"Saved dataset to {out.resolve()} (n={len(df)})")

if __name__ == "__main__":
    save_dataset()