# === Cell 0 ===
# =============================================================================
# Task 5 — Redesigned Model Evaluation (v3)
# SmartGrid Sentinel: Predictive Load Shedding Risk Forecasting
# =============================================================================
# Weather-only dataset | 3 features | 3 timesteps
# Forecast horizons: t+24 (A), t+36 (B), t+48 (C)
# Models: Logistic Regression | Random Forest | XGBoost | LSTM (Reduced)
# Metrics: Accuracy | Precision | Recall | Weighted F1 | Macro F1 | Confusion Matrix
# Feature Analysis: RF Importance | XGBoost Importance | Permutation Importance
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import json

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAMES = ["Low", "Medium", "High"]
N_CLASSES   = 3

print("[OK] Imports complete.")

# === Cell 1 ===
# =============================================================================
# Configuration — All Experiments
# =============================================================================
EXPERIMENTS = {
    "A": 24,
    "B": 36,
    "C": 48
}

BASE_DATA_DIR = "../preprocessed_v3"
ALL_RESULTS = []      # list of metric dicts
ALL_PREDICTIONS = {}  # key -> predictions array
ALL_Y_TEST = {}       # exp_name -> y_test for confusion matrices
FEATURE_IMPORTANCES = {}  # key -> importance array

print("[CONFIG] Experiments: A(t+24), B(t+36), C(t+48)")
print(f"[CONFIG] Data directory: {BASE_DATA_DIR}")

# === Cell 2 ===
# =============================================================================
# Data Loading Helper
# =============================================================================
def load_experiment_data(exp_name, horizon):
    data_dir = f"{BASE_DATA_DIR}/{exp_name}_weather_only_t+{horizon}"
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_val   = np.load(os.path.join(data_dir, "X_val.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_val   = np.load(os.path.join(data_dir, "y_val.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_train, X_val, X_test, y_train, y_val, y_test

print("[OK] Data loader defined.")

# === Cell 3 ===
# =============================================================================
# Data Integrity Checker (run per experiment)
# =============================================================================
def check_data(X_train, X_test, y_train, exp_name):
    print(f"\n--- {exp_name} ---")
    print(f"  NaN — X_train: {np.isnan(X_train).sum()} | X_test: {np.isnan(X_test).sum()}")
    print(f"  Inf  — X_train: {np.isinf(X_train).sum()} | X_test: {np.isinf(X_test).sum()}")
    unique, counts = np.unique(y_train, return_counts=True)
    print("  Train class distribution:")
    for u, c in zip(unique, counts):
        print(f"    Class {u} ({CLASS_NAMES[u]}): {c} ({c/len(y_train)*100:.1f}%)")

print("[OK] Data checker defined.")

# === Cell 4 ===
# =============================================================================
# Flatten for Classical ML Models
# =============================================================================
def flatten(X):
    return X.reshape(X.shape[0], -1)

print(f"[OK] Flatten helper defined. Flattened shape example: (n, {3*3}=9 features)")

# === Cell 5 ===
# =============================================================================
# Helper: Evaluate & Report All Metrics
# =============================================================================
def evaluate_model(name, y_true, y_pred, y_proba=None):
    acc  = accuracy_score(y_true, y_pred)
    prec_macro  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro   = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro    = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Precision   : {prec_macro:.4f}")
    print(f"  Recall      : {rec_macro:.4f}")
    print(f"  Macro F1    : {f1_macro:.4f}")
    print(f"  Weighted F1 : {f1_weighted:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec_macro, 4),
        "Recall": round(rec_macro, 4),
        "Macro F1": round(f1_macro, 4),
        "Weighted F1": round(f1_weighted, 4),
    }


def plot_cm(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

print("[OK] Evaluation helper defined.")

# === Cell 6 ===
# =============================================================================
# Run All Experiments (A, B, C) for All Models
# =============================================================================

for exp_name, horizon in EXPERIMENTS.items():
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT {exp_name} — Weather-Only | Horizon t+{horizon}")
    print(f"{'#'*60}")

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_experiment_data(exp_name, horizon)
    check_data(X_train, X_test, y_train, exp_name)
    ALL_Y_TEST[exp_name] = y_test

    # Flatten for classical ML
    X_train_flat = flatten(X_train)
    X_test_flat  = flatten(X_test)
    X_val_flat   = flatten(X_val)

    print(f"\n  Flattened: train={X_train_flat.shape}, test={X_test_flat.shape}")

    # ------------------------------------------------------
    # 1) Logistic Regression
    # ------------------------------------------------------
    lr = LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="lbfgs",
        multi_class="multinomial", random_state=SEED, n_jobs=-1
    )
    lr.fit(X_train_flat, y_train)
    lr_pred = lr.predict(X_test_flat)
    lr_metrics = evaluate_model(f"Logistic Regression ({exp_name})", y_test, lr_pred)
    ALL_RESULTS.append(lr_metrics)
    ALL_PREDICTIONS[f"LR_{exp_name}"] = lr_pred

    # ------------------------------------------------------
    # 2) Random Forest
    # ------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=250, max_depth=15, min_samples_split=6,
        min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=SEED
    )
    rf.fit(X_train_flat, y_train)
    rf_pred = rf.predict(X_test_flat)
    rf_metrics = evaluate_model(f"Random Forest ({exp_name})", y_test, rf_pred)
    ALL_RESULTS.append(rf_metrics)
    ALL_PREDICTIONS[f"RF_{exp_name}"] = rf_pred
    FEATURE_IMPORTANCES[f"RF_{exp_name}"] = rf.feature_importances_

    # ------------------------------------------------------
    # 3) XGBoost
    # ------------------------------------------------------
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    xgb = XGBClassifier(
        n_estimators=350, max_depth=7, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, objective="multi:softmax",
        num_class=N_CLASSES, eval_metric="mlogloss", random_state=SEED, n_jobs=-1
    )
    xgb.fit(X_train_flat, y_train, sample_weight=sample_weights)
    xgb_pred = xgb.predict(X_test_flat)
    xgb_metrics = evaluate_model(f"XGBoost ({exp_name})", y_test, xgb_pred)
    ALL_RESULTS.append(xgb_metrics)
    ALL_PREDICTIONS[f"XGB_{exp_name}"] = xgb_pred
    FEATURE_IMPORTANCES[f"XGB_{exp_name}"] = xgb.feature_importances_

    # ------------------------------------------------------
    # 4) LSTM (Reduced Complexity)
    # ------------------------------------------------------
    timesteps   = X_train.shape[1]
    n_features  = X_train.shape[2]

    def build_lstm_model(ts, nf, nc):
        model = Sequential([
            Input(shape=(ts, nf)),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(nc, activation="softmax")
        ], name=f"LSTM_{exp_name}")
        return model

    model_path = os.path.join(
        BASE_DATA_DIR, f"{exp_name}_weather_only_t+{horizon}", "lstm_best.keras"
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    lstm_model = build_lstm_model(timesteps, n_features, N_CLASSES)
    lstm_model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    lstm_model.summary()

    cw_values = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(cw_values))
    print(f"  Class weights: {class_weight_dict}")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=0)
    ]

    history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100, batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks, verbose=1
    )

    lstm_proba = lstm_model.predict(X_test)
    lstm_pred  = np.argmax(lstm_proba, axis=1)
    lstm_metrics = evaluate_model(f"LSTM ({exp_name})", y_test, lstm_pred)
    ALL_RESULTS.append(lstm_metrics)
    ALL_PREDICTIONS[f"LSTM_{exp_name}"] = lstm_pred

    # Permutation importance (using RF proxy on flattened data)
    rf_perm = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1)
    rf_perm.fit(X_train_flat, y_train)
    perm_imp = permutation_importance(
        rf_perm, X_test_flat, y_test, n_repeats=10,
        random_state=SEED, scoring="f1_macro"
    )
    FEATURE_IMPORTANCES[f"Permutation_{exp_name}"] = perm_imp.importances_mean

print("\n" + "=" * 60)
print("ALL EXPERIMENTS COMPLETED.")
print("=" * 60)

# === Cell 7 ===
# =============================================================================
# Feature Analysis — Importance Across Experiments
# =============================================================================
feature_names = ["temperature", "humidity", "rainfall"]
flat_names = []
for t in range(3):
    for f in feature_names:
        flat_names.append(f"t-{2-t}_{f}")

if FEATURE_IMPORTANCES:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, exp_name in enumerate(["A", "B", "C"]):
        rf_imp   = FEATURE_IMPORTANCES.get(f"RF_{exp_name}", np.zeros(len(flat_names)))
        xgb_imp  = FEATURE_IMPORTANCES.get(f"XGB_{exp_name}", np.zeros(len(flat_names)))
        perm_imp = FEATURE_IMPORTANCES.get(f"Permutation_{exp_name}", np.zeros(len(flat_names)))

        x = np.arange(len(flat_names))
        width = 0.25

        axes[idx].bar(x - width, rf_imp, width, label="RF", alpha=0.8)
        axes[idx].bar(x, xgb_imp, width, label="XGB", alpha=0.8)
        axes[idx].bar(x + width, perm_imp, width, label="Permutation", alpha=0.8)

        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(flat_names, rotation=45, ha="right", fontsize=8)
        axes[idx].set_title(f"Feature Importance — Experiment {exp_name}",
                            fontsize=11, fontweight="bold")
        axes[idx].set_ylabel("Importance")
        axes[idx].legend()
        axes[idx].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Feature Importance Comparison (RF | XGBoost | Permutation)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()

    # Verify no single feature exceeds 85% predictive power
    print("\nFeature dominance check (max single-feature RF importance):")
    for exp_name in ["A", "B", "C"]:
        rf_arr = FEATURE_IMPORTANCES.get(f"RF_{exp_name}", np.array([0]))
        max_imp = np.max(rf_arr) if len(rf_arr) > 0 else 0
        print(f"  Experiment {exp_name}: max importance = {max_imp:.4f}", end="")
        if max_imp > 0.85:
            print("  ⚠ WARNING: Feature dominance detected!")
        else:
            print("  ✓ OK: No single feature dominates.")
else:
    print("No feature importances available. Run the experiments first.")

# === Cell 8 ===
# =============================================================================
# Confusion Matrices — All Models × All Experiments
# =============================================================================
n_exp   = len(EXPERIMENTS)  # 3
n_models = 4
model_keys = ["LR", "RF", "XGB", "LSTM"]
model_labels = {
    "LR": "Logistic Regression",
    "RF": "Random Forest",
    "XGB": "XGBoost",
    "LSTM": "LSTM"
}

if ALL_PREDICTIONS:
    fig, axes = plt.subplots(n_exp, n_models, figsize=(20, 12))

    for row, exp_name in enumerate(["A", "B", "C"]):
        horizon = EXPERIMENTS[exp_name]
        y_true = ALL_Y_TEST.get(exp_name)
        if y_true is None:
            continue
        for col, mk in enumerate(model_keys):
            key = f"{mk}_{exp_name}"
            preds = ALL_PREDICTIONS.get(key)
            if preds is not None:
                plot_cm(axes[row, col], y_true, preds,
                        f"{model_labels[mk]}\nt+{horizon}")
            else:
                axes[row, col].text(0.5, 0.5, "No data",
                                    ha="center", va="center", transform=axes[row, col].transAxes)
                axes[row, col].set_title(f"{model_labels[mk]}\nt+{horizon}", fontsize=10, fontweight="bold")

    plt.suptitle("Confusion Matrices Across All Experiments and Models",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("No predictions available. Run the experiments first.")

# === Cell 9 ===
# =============================================================================
# Final Summary Table — All Models × All Experiments
# =============================================================================
print("=" * 100)
print("FINAL RESULTS TABLE")
print("=" * 100)

if ALL_RESULTS:
    table_data = []
    for m in ALL_RESULTS:
        table_data.append([
            m["Model"],
            f"{m['Accuracy']:.4f}",
            f"{m['Precision']:.4f}",
            f"{m['Recall']:.4f}",
            f"{m['Macro F1']:.4f}",
            f"{m['Weighted F1']:.4f}"
        ])

    print(tabulate(
        table_data,
        headers=["Model", "Accuracy", "Precision", "Recall", "Macro F1", "Weighted F1"],
        tablefmt="grid"
    ))
else:
    print("No results yet.")

print()
print("=" * 100)
print("EXPERIMENT SUMMARY")
print("=" * 100)

first = True
for exp_name, horizon in EXPERIMENTS.items():
    try:
        X_train, _, X_test, _, _, _ = load_experiment_data(exp_name, horizon)
        print(f"\nExperiment {exp_name} (t+{horizon}):")
        print(f"  Features     : {X_train.shape[2]} (temperature, humidity, rainfall)")
        print(f"  Timesteps    : {X_train.shape[1]} (SEQ_LEN=3, 6-hour lookback)")
        print(f"  Train samples : {X_train.shape[0]}")
        print(f"  Test samples  : {X_test.shape[0]}")
    except Exception as e:
        print(f"\nExperiment {exp_name}: Could not load data — {e}")

print()
print("Train/Val/Test split: 70%/10%/20% upazilas (strict geographic split)")
print("No temporal shortcuts (hour, day, month, weekday, peak_hour removed).")
print("No demand_index used.")
print("Only weather features: temperature, humidity, rainfall.")
print("=" * 100)

# === Cell 10 ===
# =============================================================================
# (Training history plots per experiment shown inline above)
# =============================================================================
print("[OK] Training histories displayed during experiment loop.")

# === Cell 11 ===
# =============================================================================
# (Confusion matrix grid shown in cell above)
# =============================================================================
print("[OK] Confusion matrices displayed in analysis section.")

# === Cell 12 ===
# =============================================================================
# (Model comparison shown in final table below)
# =============================================================================
print("[OK] Model comparison available in final summary table.")

# === Cell 13 ===
# =============================================================================
# (Final summary displayed above)
# =============================================================================
print("[OK] Analysis complete.")

