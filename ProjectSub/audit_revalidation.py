import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
import os
import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# STEP 1: LOAD RAW DATA AND VERIFY LEAKAGE MECHANISM
# ============================================================
print("="*70)
print("STEP 1: VERIFYING LEAKAGE MECHANISM")
print("="*70)

df = pd.read_csv('smart_grid_dataset_sylhet.csv')
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check how risk_level relates to demand_index
print("\n--- demand_index thresholds by risk_level ---")
for risk in ['Low', 'Medium', 'High']:
    subset = df[df['risk_level'] == risk]['demand_index']
    print(f"{risk}: min={subset.min():.4f}, max={subset.max():.4f}, mean={subset.mean():.4f}")

# Verify if risk_level can be perfectly predicted from demand_index thresholds
print("\n--- Threshold-based prediction test ---")
def predict_from_demand(d):
    if d < 0.45:
        return 'Low'
    elif d < 0.70:
        return 'Medium'
    else:
        return 'High'

df['predicted_risk'] = df['demand_index'].apply(predict_from_demand)
threshold_acc = accuracy_score(df['risk_level'], df['predicted_risk'])
print(f"Accuracy using demand_index thresholds alone: {threshold_acc:.4f}")
print(f"Misclassified samples: {(df['risk_level'] != df['predicted_risk']).sum()}")

# Show any misclassifications
misclassified = df[df['risk_level'] != df['predicted_risk']]
if len(misclassified) > 0:
    print("\nMisclassified samples:")
    print(misclassified[['demand_index', 'risk_level', 'predicted_risk']].head(10))

# ============================================================
# STEP 2: REPLICATE PREPROCESSING PIPELINE
# ============================================================
print("\n" + "="*70)
print("STEP 2: REPLICATING PREPROCESSING")
print("="*70)

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(['upazila', 'datetime']).reset_index(drop=True)

risk_map = {"Low": 0, "Medium": 1, "High": 2}
df["risk_encoded"] = df["risk_level"].map(risk_map)

# Feature engineering (same as notebook)
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["weekday"] = df["datetime"].dt.weekday
df["peak_hour"] = df["hour"].apply(lambda h: 1 if (6 <= h <= 10) or (18 <= h <= 22) else 0)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

all_features = [
    "temperature", "humidity", "rainfall", "demand_index",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "day", "month", "peak_hour",
]

# ============================================================
# STEP 3: CREATE THREE FEATURE SETS
# ============================================================
feature_sets = {
    'A_demand_only': ['demand_index'],
    'B_no_demand': [f for f in all_features if f != 'demand_index'],
    'C_all_features': all_features
}

# ============================================================
# STEP 4: PER-UPZILA CHRONOLOGICAL SPLIT
# ============================================================
TRAIN_RATIO = 0.80
train_frames = []
test_frames = []

for upazila_name, group in df.groupby("upazila", sort=False):
    group = group.sort_values("datetime").reset_index(drop=True)
    split_idx = int(len(group) * TRAIN_RATIO)
    train_frames.append(group.iloc[:split_idx])
    test_frames.append(group.iloc[split_idx:])

df_train = pd.concat(train_frames).reset_index(drop=True)
df_test = pd.concat(test_frames).reset_index(drop=True)

print(f"\nTrain rows: {len(df_train)} | Test rows: {len(df_test)}")

# ============================================================
# STEP 5: BUILD SEQUENCES FOR EACH FEATURE SET
# ============================================================
SEQ_LEN = 6

def create_sequences_1step_ahead(X_scaled, y_raw, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i : i + seq_len])
        y_seq.append(y_raw[i + seq_len])
    return np.array(X_seq), np.array(y_seq)

def build_sequences_upazila_wise(df_split, X_scaled_all, y_raw_all, seq_len):
    X_all, y_all = [], []
    for _, group in df_split.groupby("upazila", sort=False):
        idx = group.index
        X_up = X_scaled_all[idx]
        y_up = y_raw_all[idx]
        X_s, y_s = create_sequences_1step_ahead(X_up, y_up, seq_len)
        X_all.extend(X_s)
        y_all.extend(y_s)
    return np.array(X_all), np.array(y_all)

results = {}

for set_name, features in feature_sets.items():
    print(f"\n{'='*70}")
    print(f"PROCESSING FEATURE SET: {set_name}")
    print(f"Features: {features}")
    print(f"{'='*70}")
    
    # Extract features
    X_train_raw = df_train[features].values
    X_test_raw = df_test[features].values
    y_train_raw = df_train["risk_encoded"].values
    y_test_raw = df_test["risk_encoded"].values
    
    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Build sequences
    X_train_seq, y_train_seq = build_sequences_upazila_wise(
        df_train, X_train_scaled, y_train_raw, SEQ_LEN
    )
    X_test_seq, y_test_seq = build_sequences_upazila_wise(
        df_test, X_test_scaled, y_test_raw, SEQ_LEN
    )
    
    # Validation split (last 15% of train)
    VAL_RATIO = 0.15
    val_split_idx = int(len(X_train_seq) * (1 - VAL_RATIO))
    X_val_seq, y_val_seq = X_train_seq[val_split_idx:], y_train_seq[val_split_idx:]
    X_train_seq, y_train_seq = X_train_seq[:val_split_idx], y_train_seq[:val_split_idx]
    
    print(f"Shapes: X_train={X_train_seq.shape}, X_val={X_val_seq.shape}, X_test={X_test_seq.shape}")
    
    # ============================================================
    # STEP 6: TRAIN LSTM MODEL
    # ============================================================
    TIMESTEPS = X_train_seq.shape[1]
    N_FEATURES = X_train_seq.shape[2]
    N_CLASSES = 3
    
    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    cw_values = compute_class_weight("balanced", classes=np.unique(y_train_seq), y=y_train_seq)
    class_weight_dict = dict(enumerate(cw_values))
    print(f"Class weights: {class_weight_dict}")
    
    def build_lstm_model(timesteps, n_features, n_classes):
        model = Sequential([
            Input(shape=(timesteps, n_features)),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(n_classes, activation="softmax")
        ])
        return model
    
    model = build_lstm_model(TIMESTEPS, N_FEATURES, N_CLASSES)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model_save_path = f"preprocessed_fixed/lstm_{set_name}_best.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=0),
        ModelCheckpoint(model_save_path, monitor="val_loss", save_best_only=True, verbose=0)
    ]
    
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=80,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=0
    )
    
    # Predictions
    lstm_proba = model.predict(X_test_seq, verbose=0)
    lstm_pred = np.argmax(lstm_proba, axis=1)
    
    # Metrics
    acc = accuracy_score(y_test_seq, lstm_pred)
    prec = precision_score(y_test_seq, lstm_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test_seq, lstm_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test_seq, lstm_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_test_seq, lstm_pred, average="macro", zero_division=0)
    
    # Per-class metrics
    cm = confusion_matrix(y_test_seq, lstm_pred)
    cr = classification_report(y_test_seq, lstm_pred, target_names=['Low', 'Medium', 'High'], output_dict=True)
    
    results[set_name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_weighted': f1_w,
        'f1_macro': f1_m,
        'confusion_matrix': cm,
        'classification_report': cr,
        'history': history,
        'model': model,
        'X_train': X_train_seq,
        'X_val': X_val_seq,
        'X_test': X_test_seq,
        'y_train': y_train_seq,
        'y_val': y_val_seq,
        'y_test': y_test_seq,
        'y_pred': lstm_pred,
        'y_proba': lstm_proba
    }
    
    print(f"\n--- RESULTS for {set_name} ---")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print(f"F1 (weighted): {f1_w:.4f}")
    print(f"F1 (macro):    {f1_m:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_test_seq, lstm_pred, target_names=['Low', 'Medium', 'High']))

# ============================================================
# STEP 7: COMPARISON TABLE
# ============================================================
print("\n" + "="*70)
print("COMPARISON TABLE")
print("="*70)
print(f"{'Feature Set':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1(w)':>10} {'F1(m)':>10}")
print("-"*70)
for set_name, res in results.items():
    print(f"{set_name:<20} {res['accuracy']:>10.4f} {res['precision']:>10.4f} {res['recall']:>10.4f} {res['f1_weighted']:>10.4f} {res['f1_macro']:>10.4f}")

# ============================================================
# STEP 8: RANDOM FOREST AND XGBOOST FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)

# Use all features for this analysis
X_train_all = df_train[all_features].values
X_test_all = df_test[all_features].values
y_train_all = df_train["risk_encoded"].values
y_test_all = df_test["risk_encoded"].values

# Scale
scaler_all = MinMaxScaler()
X_train_all_scaled = scaler_all.fit_transform(X_train_all)
X_test_all_scaled = scaler_all.transform(X_test_all)

# Random Forest
print("\n--- Random Forest Feature Importance ---")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=SEED, n_jobs=-1)
rf.fit(X_train_all_scaled, y_train_all)
rf_importance = pd.DataFrame({
    'feature': all_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(rf_importance.to_string(index=False))

# XGBoost
print("\n--- XGBoost Feature Importance ---")
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, 
                    subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                    use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train_all_scaled, y_train_all)
xgb_importance = pd.DataFrame({
    'feature': all_features,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=False)
print(xgb_importance.to_string(index=False))

# Performance without demand_index using sklearn models
print("\n--- Performance Without demand_index (Sklearn Models) ---")
features_no_demand = [f for f in all_features if f != 'demand_index']
X_train_no_demand = df_train[features_no_demand].values
X_test_no_demand = df_test[features_no_demand].values
scaler_no_demand = MinMaxScaler()
X_train_no_demand_scaled = scaler_no_demand.fit_transform(X_train_no_demand)
X_test_no_demand_scaled = scaler_no_demand.transform(X_test_no_demand)

# RF without demand_index
rf_no_demand = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=SEED, n_jobs=-1)
rf_no_demand.fit(X_train_no_demand_scaled, y_train_all)
rf_no_demand_pred = rf_no_demand.predict(X_test_no_demand_scaled)
print(f"RF without demand_index: Accuracy={accuracy_score(y_test_all, rf_no_demand_pred):.4f}, "
      f"F1(w)={f1_score(y_test_all, rf_no_demand_pred, average='weighted'):.4f}")

# RF with demand_index
rf_with_demand_pred = rf.predict(X_test_all_scaled)
print(f"RF with demand_index:    Accuracy={accuracy_score(y_test_all, rf_with_demand_pred):.4f}, "
      f"F1(w)={f1_score(y_test_all, rf_with_demand_pred, average='weighted'):.4f}")

# ============================================================
# STEP 9: TRAINING HISTORY PLOTS
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, (set_name, res) in enumerate(results.items()):
    ax = axes[idx // 3, idx % 3]
    ax.plot(res['history'].history['accuracy'], label='Train')
    ax.plot(res['history'].history['val_accuracy'], label='Validation')
    ax.set_title(f'{set_name} - Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
# Hide unused subplot
axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('training_history_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved training_history_comparison.png")

# ============================================================
# STEP 10: CONFUSION MATRIX COMPARISON
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
class_names = ['Low', 'Medium', 'High']
for idx, (set_name, res) in enumerate(results.items()):
    sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[idx])
    axes[idx].set_title(f'{set_name}\nAccuracy: {res["accuracy"]:.4f}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=150, bbox_inches='tight')
print("Saved confusion_matrix_comparison.png")

# ============================================================
# STEP 11: GENERALIZATION ASSESSMENT
# ============================================================
print("\n" + "="*70)
print("GENERALIZATION ASSESSMENT")
print("="*70)

for set_name, res in results.items():
    train_loss = res['history'].history['loss'][-1]
    val_loss = res['history'].history['val_loss'][-1]
    train_acc = res['history'].history['accuracy'][-1]
    val_acc = res['history'].history['val_accuracy'][-1]
    test_acc = res['accuracy']
    
    gap_train_val = train_acc - val_acc
    gap_val_test = val_acc - test_acc
    
    print(f"\n{set_name}:")
    print(f"  Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
    print(f"  Gap (Train-Val): {gap_train_val:.4f} | Gap (Val-Test): {gap_val_test:.4f}")
    
    if gap_train_val > 0.05 and gap_val_test > 0.02:
        print(f"  -> OVERFITTING DETECTED")
    elif gap_train_val < 0.02 and gap_val_test < 0.02:
        print(f"  -> GOOD GENERALIZATION")
    else:
        print(f"  -> MILD GAP")

print("\n" + "="*70)
print("AUDIT COMPLETE")
print("="*70)