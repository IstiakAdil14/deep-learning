"""
Generate preprocessed_v3 datasets for Task4.
"""
import os
import json
import random
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

df = pd.read_csv("smart_grid_dataset_sylhet.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values(["upazila", "datetime"]).reset_index(drop=True)

risk_map = {"Low": 0, "Medium": 1, "High": 2}
df["risk_encoded"] = df["risk_level"].map(risk_map)

feature_cols = ["temperature", "humidity", "rainfall"]
target_col = "risk_encoded"
SEQ_LEN = 3

os.makedirs("preprocessed_v3", exist_ok=True)

upazilas = df["upazila"].unique()
train_upazilas, temp_upazilas = train_test_split(upazilas, train_size=0.7, random_state=SEED)
val_upazilas, test_upazilas = train_test_split(temp_upazilas, test_size=0.667, random_state=SEED)

df_train = df[df["upazila"].isin(train_upazilas)].copy().reset_index(drop=True)
df_val = df[df["upazila"].isin(val_upazilas)].copy().reset_index(drop=True)
df_test = df[df["upazila"].isin(test_upazilas)].copy().reset_index(drop=True)

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_raw = df_train[feature_cols].values
X_val_raw = df_val[feature_cols].values
X_test_raw = df_test[feature_cols].values

X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)

y_train_raw = df_train[target_col].values
y_val_raw = df_val[target_col].values
y_test_raw = df_test[target_col].values

def create_sequences(X_scaled, y_raw, seq_len, horizon):
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - horizon):
        X_seq.append(X_scaled[i:i + seq_len])
        y_seq.append(y_raw[i + horizon])
    return np.array(X_seq), np.array(y_seq)

def build_sequences(df_split, X_scaled_all, y_raw_all, seq_len, horizon):
    X_all, y_all = [], []
    for _, group in df_split.groupby("upazila", sort=False):
        idx = group.index
        X_up = X_scaled_all[idx]
        y_up = y_raw_all[idx]
        X_s, y_s = create_sequences(X_up, y_up, seq_len, horizon)
        X_all.extend(X_s)
        y_all.extend(y_s)
    return np.array(X_all), np.array(y_all)

for exp_name, horizon in [("A", 24), ("B", 36), ("C", 48)]:
    save_dir = f"preprocessed_v3/{exp_name}_weather_only_t+{horizon}"
    os.makedirs(save_dir, exist_ok=True)

    X_train_seq, y_train_seq = build_sequences(df_train, X_train_scaled, y_train_raw, SEQ_LEN, horizon)
    X_val_seq, y_val_seq = build_sequences(df_val, X_val_scaled, y_val_raw, SEQ_LEN, horizon)
    X_test_seq, y_test_seq = build_sequences(df_test, X_test_scaled, y_test_raw, SEQ_LEN, horizon)

    np.save(os.path.join(save_dir, "X_train.npy"), X_train_seq)
    np.save(os.path.join(save_dir, "X_val.npy"), X_val_seq)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test_seq)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train_seq)
    np.save(os.path.join(save_dir, "y_val.npy"), y_val_seq)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test_seq)

    joblib.dump(scaler, os.path.join(save_dir, "feature_scaler.pkl"))

    report = {
        "experiment": exp_name,
        "horizon": horizon,
        "seq_len": SEQ_LEN,
        "features": feature_cols,
        "train_upazilas": train_upazilas.tolist(),
        "val_upazilas": val_upazilas.tolist(),
        "test_upazilas": test_upazilas.tolist(),
        "train_timestamps": df_train["datetime"].unique().tolist(),
        "val_timestamps": df_val["datetime"].unique().tolist(),
        "test_timestamps": df_test["datetime"].unique().tolist(),
        "X_train_shape": X_train_seq.shape,
        "X_val_shape": X_val_seq.shape,
        "X_test_shape": X_test_seq.shape,
    }

    with open(os.path.join(save_dir, "preprocessing_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[{exp_name}] Horizon t+{horizon}: train={X_train_seq.shape}, val={X_val_seq.shape}, test={X_test_seq.shape}")

print("\n[OK] preprocessed_v3 generated successfully.")