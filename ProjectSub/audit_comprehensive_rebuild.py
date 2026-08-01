"""
SmartGrid Sentinel — Comprehensive Pipeline Rebuild
================================================================
Modifies preprocessing and training notebooks to create a scientifically
defensible forecasting system with:
- 3 dataset versions (A: Weather Only, B: Weather+Temporal, C: Full)
- 3 forecast horizons (t+6, t+12, t+24)
- Geographic generalization (train on seen upazilas, test on unseen)
- 4 model types (LR, RF, XGBoost, LSTM)
- Full audit trail
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = 'preprocessed_final'
os.makedirs(BASE_DIR, exist_ok=True)

# ============================================================
# LOAD RAW DATA
# ============================================================
print("="*70)
print("SMARTGRID SENTINEL — COMPREHENSIVE PIPELINE REBUILD")
print("="*70)

df = pd.read_csv('smart_grid_dataset_sylhet.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(['upazila', 'datetime']).reset_index(drop=True)

print(f"\nRaw dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Upazilas: {df['upazila'].nunique()}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# ============================================================
# FEATURE ENGINEERING (shared across all versions)
# ============================================================
print("\n" + "="*70)
print("FEATURE ENGINEERING")
print("="*70)

df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["weekday"] = df["datetime"].dt.weekday
df["peak_hour"] = df["hour"].apply(lambda h: 1 if (6 <= h <= 10) or (18 <= h <= 22) else 0)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
df["risk_encoded"] = df["risk_level"].map({"Low": 0, "Medium": 1, "High": 2})

# Feature sets
FEATURE_SETS = {
    'A_weather_only': ['temperature', 'humidity', 'rainfall'],
    'B_weather_temporal': ['temperature', 'humidity', 'rainfall',
                           'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos'],
    'C_full': ['temperature', 'humidity', 'rainfall',
               'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
               'peak_hour', 'day', 'month']
}

FORECAST_HORIZONS = {'t+6': 6, 't+12': 12, 't+24': 24}
SEQ_LEN = 6  # Always use 6 timesteps lookback

# ============================================================
# EXPERIMENT 1: GEOGRAPHIC GENERALIZATION
# ============================================================
print("\n" + "="*70)
print("EXPERIMENT 1: GEOGRAPHIC GENERALIZATION")
print("="*70)

upazilas = df['upazila'].unique().tolist()
np.random.seed(SEED)
np.random.shuffle(upazilas)

n_train_upazilas = int(0.8 * len(upazilas))
train_upazilas = upazilas[:n_train_upazilas]
test_upazilas = upazilas[n_train_upazilas:]

print(f"\nTrain upazilas ({len(train_upazilas)}): {train_upazilas[:5]}...")
print(f"Test upazilas ({len(test_upazilas)}): {test_upazilas[:5]}...")

geo_train = df[df['upazila'].isin(train_upazilas)].copy()
geo_test = df[df['upazila'].isin(test_upazilas)].copy()

print(f"\nGeographic split: Train={len(geo_train)}, Test={len(geo_test)}")

# ============================================================
# BUILD ALL DATASETS
# ============================================================
print("\n" + "="*70)
print("BUILDING ALL DATASET VERSIONS")
print("="*70)

def create_sequences_horizon(X_scaled, y_raw, seq_len, horizon):
    """Create sequences with specified forecast horizon."""
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len - horizon + 1):
        X_seq.append(X_scaled[i : i + seq_len])
        y_seq.append(y_raw[i + seq_len + horizon - 1])
    return np.array(X_seq), np.array(y_seq)

def build_sequences_upazila_wise_horizon(df_split, X_scaled_all, y_raw_all, seq_len, horizon):
    X_all, y_all = [], []
    for _, group in df_split.groupby("upazila", sort=False):
        idx = group.index
        X_up = X_scaled_all[idx]
        y_up = y_raw_all[idx]
        X_s, y_s = create_sequences_horizon(X_up, y_up, seq_len, horizon)
        X_all.extend(X_s)
        y_all.extend(y_s)
    return np.array(X_all), np.array(y_all)

# Standard chronological split (for main experiments)
TRAIN_RATIO = 0.80
train_frames = []
test_frames = []
for upazila_name, group in df.groupby("upazila", sort=False):
    group = group.sort_values("datetime").reset_index(drop=True)
    split_idx = int(len(group) * TRAIN_RATIO)
    train_frames.append(group.iloc[:split_idx])
    test_frames.append(group.iloc[split_idx:])
df_train_chrono = pd.concat(train_frames).reset_index(drop=True)
df_test_chrono = pd.concat(test_frames).reset_index(drop=True)

# Build datasets for each version, horizon, and split type
all_datasets = {}

for version_name, features in FEATURE_SETS.items():
    for horizon_name, horizon_steps in FORECAST_HORIZONS.items():
        # Chronological split
        X_train_raw = df_train_chrono[features].values
        X_test_raw = df_test_chrono[features].values
        y_train_raw = df_train_chrono["risk_encoded"].values
        y_test_raw = df_test_chrono["risk_encoded"].values
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        
        X_train_seq, y_train_seq = build_sequences_upazila_wise_horizon(
            df_train_chrono, X_train_scaled, y_train_raw, SEQ_LEN, horizon_steps
        )
        X_test_seq, y_test_seq = build_sequences_upazila_wise_horizon(
            df_test_chrono, X_test_scaled, y_test_raw, SEQ_LEN, horizon_steps
        )
        
        # Validation split
        VAL_RATIO = 0.15
        val_split_idx = int(len(X_train_seq) * (1 - VAL_RATIO))
        X_val_seq, y_val_seq = X_train_seq[val_split_idx:], y_train_seq[val_split_idx:]
        X_train_seq, y_train_seq = X_train_seq[:val_split_idx], y_train_seq[:val_split_idx]
        
        dataset_name = f"{version_name}_{horizon_name}"
        all_datasets[dataset_name] = {
            'X_train': X_train_seq, 'y_train': y_train_seq,
            'X_val': X_val_seq, 'y_val': y_val_seq,
            'X_test': X_test_seq, 'y_test': y_test_seq,
            'features': features, 'horizon': horizon_steps,
            'scaler': scaler, 'version': version_name, 'horizon_name': horizon_name
        }
        
        print(f"  {dataset_name}: X_train={X_train_seq.shape}, X_test={X_test_seq.shape}")

# Build geographic split dataset (using Version C, t+6 as example)
print("\n--- Geographic split dataset (Version C, t+6) ---")
geo_train = geo_train.reset_index(drop=True)
geo_test = geo_test.reset_index(drop=True)

X_train_raw = geo_train[FEATURE_SETS['C_full']].values
X_test_raw = geo_test[FEATURE_SETS['C_full']].values
y_train_raw = geo_train["risk_encoded"].values
y_test_raw = geo_test["risk_encoded"].values

scaler_geo = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler_geo.fit_transform(X_train_raw)
X_test_scaled = scaler_geo.transform(X_test_raw)

X_train_geo, y_train_geo = build_sequences_upazila_wise_horizon(
    geo_train, X_train_scaled, y_train_raw, SEQ_LEN, 6
)
X_test_geo, y_test_geo = build_sequences_upazila_wise_horizon(
    geo_test, X_test_scaled, y_test_raw, SEQ_LEN, 6
)

val_split_idx = int(len(X_train_geo) * (1 - VAL_RATIO))
X_val_geo, y_val_geo = X_train_geo[val_split_idx:], y_train_geo[val_split_idx:]
X_train_geo, y_train_geo = X_train_geo[:val_split_idx], y_train_geo[:val_split_idx]

all_datasets['C_full_t+6_geo'] = {
    'X_train': X_train_geo, 'y_train': y_train_geo,
    'X_val': X_val_geo, 'y_val': y_val_geo,
    'X_test': X_test_geo, 'y_test': y_test_geo,
    'features': FEATURE_SETS['C_full'], 'horizon': 6,
    'scaler': scaler_geo, 'version': 'C_full', 'horizon_name': 't+6',
    'split_type': 'geographic'
}

print(f"  C_full_t+6_geo: X_train={X_train_geo.shape}, X_test={X_test_geo.shape}")

# Save all datasets
for name, data in all_datasets.items():
    save_dir = os.path.join(BASE_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/X_train.npy', data['X_train'])
    np.save(f'{save_dir}/X_val.npy', data['X_val'])
    np.save(f'{save_dir}/X_test.npy', data['X_test'])
    np.save(f'{save_dir}/y_train.npy', data['y_train'])
    np.save(f'{save_dir}/y_val.npy', data['y_val'])
    np.save(f'{save_dir}/y_test.npy', data['y_test'])
    joblib.dump(data['scaler'], f'{save_dir}/scaler.pkl')
    with open(f'{save_dir}/features.txt', 'w') as f:
        f.write('\n'.join(data['features']))

print(f"\nSaved all datasets to {BASE_DIR}/")

# ============================================================
# MODEL TRAINING AND EVALUATION
# ============================================================
print("\n" + "="*70)
print("MODEL TRAINING AND EVALUATION")
print("="*70)

def build_lstm_model(timesteps, n_features, n_classes):
    model = Sequential([
        Input(shape=(timesteps, n_features)),
        LSTM(128, return_sequences=True), Dropout(0.3),
        LSTM(64, return_sequences=True), Dropout(0.3),
        LSTM(32), Dropout(0.2),
        Dense(64, activation='relu'), Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])
    return model

def train_and_evaluate(model_name, dataset_name, data, use_lstm=False):
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    if use_lstm:
        timesteps = X_train.shape[1]
        n_features = X_train.shape[2]
        model = build_lstm_model(timesteps, n_features, 3)
        model.compile(optimizer=Adam(learning_rate=1e-3),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        cw_values = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(cw_values))
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0),
            ModelCheckpoint(f'{BASE_DIR}/{dataset_name}_{model_name}_best.keras',
                          monitor='val_loss', save_best_only=True, verbose=0)
        ]
        
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                          epochs=80, batch_size=64, class_weight=class_weight_dict,
                          callbacks=callbacks, verbose=0)
        
        proba = model.predict(X_test, verbose=0)
        pred = np.argmax(proba, axis=1)
        
        return {
            'model': model, 'history': history, 'y_pred': pred, 'y_proba': proba,
            'train_acc': history.history['accuracy'][-1],
            'val_acc': history.history['val_accuracy'][-1]
        }
    else:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        if model_name == 'LogisticRegression':
            model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=SEED, n_jobs=-1)
        elif model_name == 'XGBoost':
            model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                                use_label_encoder=False, eval_metric='mlogloss')
        
        model.fit(X_train_flat, y_train)
        pred = model.predict(X_test_flat)
        proba = model.predict_proba(X_test_flat) if hasattr(model, 'predict_proba') else None
        
        return {
            'model': model, 'y_pred': pred, 'y_proba': proba,
            'train_acc': None, 'val_acc': None
        }

results = []
all_models = {}

for dataset_name, data in all_datasets.items():
    print(f"\n--- {dataset_name} ---")
    
    for model_name in ['LogisticRegression', 'RandomForest', 'XGBoost', 'LSTM']:
        use_lstm = (model_name == 'LSTM')
        result = train_and_evaluate(model_name, dataset_name, data, use_lstm)
        
        y_test = data['y_test']
        acc = accuracy_score(y_test, result['y_pred'])
        prec = precision_score(y_test, result['y_pred'], average='weighted', zero_division=0)
        rec = recall_score(y_test, result['y_pred'], average='weighted', zero_division=0)
        f1_w = f1_score(y_test, result['y_pred'], average='weighted', zero_division=0)
        f1_m = f1_score(y_test, result['y_pred'], average='macro', zero_division=0)
        cm = confusion_matrix(y_test, result['y_pred'])
        cr = classification_report(y_test, result['y_pred'], 
                                  target_names=['Low', 'Medium', 'High'], output_dict=True)
        
        results.append({
            'dataset': dataset_name,
            'model': model_name,
            'version': data['version'],
            'horizon': data['horizon_name'],
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_weighted': f1_w,
            'f1_macro': f1_m,
            'confusion_matrix': cm,
            'classification_report': cr,
            'train_acc': result.get('train_acc'),
            'val_acc': result.get('val_acc')
        })
        
        print(f"  {model_name}: Acc={acc:.4f}, F1(w)={f1_w:.4f}, F1(m)={f1_m:.4f}")
        
        if model_name not in all_models:
            all_models[model_name] = {}
        all_models[model_name][dataset_name] = result

# ============================================================
# RESULTS SUMMARY
# ============================================================
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

results_df = pd.DataFrame(results)
summary = results_df.groupby(['model', 'version', 'horizon']).agg({
    'accuracy': 'mean',
    'f1_weighted': 'mean',
    'f1_macro': 'mean'
}).reset_index()

print("\n--- Accuracy by Model, Version, Horizon ---")
pivot = summary.pivot_table(index=['model', 'version'], columns='horizon', values='accuracy')
print(pivot.round(4))

print("\n--- Macro F1 by Model, Version, Horizon ---")
pivot_f1 = summary.pivot_table(index=['model', 'version'], columns='horizon', values='f1_macro')
print(pivot_f1.round(4))

# Best model per horizon
print("\n--- Best Model per Horizon (by Macro F1) ---")
for horizon in ['t+6', 't+12', 't+24']:
    best = summary[summary['horizon'] == horizon].loc[summary[summary['horizon'] == horizon]['f1_macro'].idxmax()]
    print(f"  {horizon}: {best['model']} ({best['version']}) - F1(m)={best['f1_macro']:.4f}")

# Best model overall
best_overall = summary.loc[summary['f1_macro'].idxmax()]
print(f"\n--- OVERALL BEST ---")
print(f"  {best_overall['model']} | {best_overall['version']} | {best_overall['horizon']} | F1(m)={best_overall['f1_macro']:.4f}")

# ============================================================
# GEOGRAPHIC GENERALIZATION RESULTS
# ============================================================
print("\n" + "="*70)
print("GEOGRAPHIC GENERALIZATION RESULTS")
print("="*70)

geo_results = results_df[results_df['dataset'] == 'C_full_t+6_geo']
print(geo_results[['model', 'accuracy', 'f1_weighted', 'f1_macro']].to_string(index=False))

# Compare with standard split
standard_results = results_df[(results_df['dataset'] == 'C_full_t+6') & (results_df['model'] == 'LSTM')]
print("\n--- Comparison: Standard vs Geographic Split (LSTM) ---")
if len(standard_results) > 0:
    std_acc = standard_results['accuracy'].values[0]
    std_f1 = standard_results['f1_macro'].values[0]
    geo_acc = geo_results[geo_results['model'] == 'LSTM']['accuracy'].values[0]
    geo_f1 = geo_results[geo_results['model'] == 'LSTM']['f1_macro'].values[0]
    print(f"  Standard split: Acc={std_acc:.4f}, F1(m)={std_f1:.4f}")
    print(f"  Geographic split: Acc={geo_acc:.4f}, F1(m)={geo_f1:.4f}")
    print(f"  Drop: {(std_acc - geo_acc):.4f} accuracy, {(std_f1 - geo_f1):.4f} F1(m)")

# ============================================================
# FEATURE IMPORTANCE (on Version C, t+6)
# ============================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE (Version C, t+6)")
print("="*70)

dataset_key = 'C_full_t+6'
data_c = all_datasets[dataset_key]
X_train_flat = data_c['X_train'].reshape(data_c['X_train'].shape[0], -1)
X_test_flat = data_c['X_test'].reshape(data_c['X_test'].shape[0], -1)
y_test = data_c['y_test']

# RF importance
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=SEED, n_jobs=-1)
rf.fit(X_train_flat, data_c['y_train'])
rf_imp = pd.DataFrame({
    'feature': FEATURE_SETS['C_full'],
    'importance': [rf.feature_importances_[i*SEQ_LEN:(i+1)*SEQ_LEN].sum() 
                   for i in range(len(FEATURE_SETS['C_full']))]
}).sort_values('importance', ascending=False)
print("\n--- Random Forest ---")
print(rf_imp.to_string(index=False))

# XGBoost importance
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                    use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train_flat, data_c['y_train'])
xgb_imp = pd.DataFrame({
    'feature': FEATURE_SETS['C_full'],
    'importance': [xgb.feature_importances_[i*SEQ_LEN:(i+1)*SEQ_LEN].sum() 
                   for i in range(len(FEATURE_SETS['C_full']))]
}).sort_values('importance', ascending=False)
print("\n--- XGBoost ---")
print(xgb_imp.to_string(index=False))

# Permutation importance
print("\n--- Permutation Importance ---")
perm_imp = permutation_importance(rf, X_test_flat, y_test, n_repeats=5, random_state=SEED, n_jobs=1)
perm_df = pd.DataFrame({
    'feature': FEATURE_SETS['C_full'],
    'importance': [perm_imp.importances_mean[i*SEQ_LEN:(i+1)*SEQ_LEN].sum() 
                   for i in range(len(FEATURE_SETS['C_full']))]
}).sort_values('importance', ascending=False)
print(perm_df.to_string(index=False))

# ============================================================
# TRAINING HISTORY PLOTS
# ============================================================
print("\n" + "="*70)
print("GENERATING PLOTS")
print("="*70)

# Plot training histories for LSTM on key configurations
configs = ['C_full_t+6', 'A_weather_only_t+6', 'C_full_t+24']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, config in enumerate(configs):
    if config in all_models.get('LSTM', {}):
        hist = all_models['LSTM'][config]['history']
        # Accuracy
        axes[0, idx].plot(hist.history['accuracy'], label='Train')
        axes[0, idx].plot(hist.history['val_accuracy'], label='Validation')
        axes[0, idx].set_title(f'{config} - Accuracy')
        axes[0, idx].set_xlabel('Epoch')
        axes[0, idx].set_ylabel('Accuracy')
        axes[0, idx].legend()
        axes[0, idx].grid(True)
        # Loss
        axes[1, idx].plot(hist.history['loss'], label='Train')
        axes[1, idx].plot(hist.history['val_loss'], label='Validation')
        axes[1, idx].set_title(f'{config} - Loss')
        axes[1, idx].set_xlabel('Epoch')
        axes[1, idx].set_ylabel('Loss')
        axes[1, idx].legend()
        axes[1, idx].grid(True)

plt.tight_layout()
plt.savefig(f'{BASE_DIR}/training_histories.png', dpi=150, bbox_inches='tight')
print(f"Saved training_histories.png")

# Confusion matrices for best configurations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
best_configs = [('C_full_t+6', 'LSTM'), ('C_full_t+6', 'XGBoost'), ('A_weather_only_t+6', 'LSTM'),
                ('C_full_t+24', 'LSTM'), ('C_full_t+6_geo', 'LSTM')]
for idx, (ds_name, model_name) in enumerate(best_configs[:5]):
    if model_name == 'LSTM':
        res = all_models[model_name][ds_name]
        y_pred = res['y_pred']
    else:
        res = all_models[model_name][ds_name]
        y_pred = res['y_pred']
    
    y_test = all_datasets[ds_name]['y_test']
    cm = confusion_matrix(y_test, y_pred)
    
    row = idx // 3
    col = idx % 3
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'],
                ax=axes[row, col])
    axes[row, col].set_title(f'{ds_name} - {model_name}')
    axes[row, col].set_xlabel('Predicted')
    axes[row, col].set_ylabel('Actual')

axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig(f'{BASE_DIR}/confusion_matrices.png', dpi=150, bbox_inches='tight')
print(f"Saved confusion_matrices.png")

# ============================================================
# FINAL COMPARISON TABLE
# ============================================================
print("\n" + "="*70)
print("FINAL COMPARISON TABLE")
print("="*70)

final_table = results_df[['model', 'version', 'horizon', 'accuracy', 'f1_weighted', 'f1_macro']].copy()
final_table = final_table.sort_values(['horizon', 'f1_macro'], ascending=[True, False])

print(final_table.to_string(index=False))

# Rank models
print("\n--- RANKING BY MACRO F1 (averaged across all versions and horizons) ---")
ranking = results_df.groupby('model')['f1_macro'].mean().sort_values(ascending=False)
for rank, (model, f1) in enumerate(ranking.items(), 1):
    print(f"  {rank}. {model}: {f1:.4f}")

# Save final results
results_df.to_csv(f'{BASE_DIR}/all_results.csv', index=False)
print(f"\nSaved all_results.csv")

# ============================================================
# ANSWER KEY QUESTIONS
# ============================================================
print("\n" + "="*70)
print("ANSWER KEY QUESTIONS")
print("="*70)

# 1. Which setup generalizes best?
# Compare geographic vs standard split for LSTM
geo_lstm = results_df[(results_df['dataset'] == 'C_full_t+6_geo') & (results_df['model'] == 'LSTM')]
std_lstm = results_df[(results_df['dataset'] == 'C_full_t+6') & (results_df['model'] == 'LSTM')]
if len(geo_lstm) > 0 and len(std_lstm) > 0:
    geo_f1 = geo_lstm['f1_macro'].values[0]
    std_f1 = std_lstm['f1_macro'].values[0]
    print(f"\n1. Geographic generalization (LSTM, C_full_t+6):")
    print(f"   Standard split F1(m): {std_f1:.4f}")
    print(f"   Geographic split F1(m): {geo_f1:.4f}")
    print(f"   Generalization gap: {std_f1 - geo_f1:.4f}")

# 2. Which features matter most?
print(f"\n2. Most important features (XGBoost, C_full_t+6):")
print(xgb_imp.head(5).to_string(index=False))

# 3. Is dataset too easy?
max_acc = results_df['accuracy'].max()
print(f"\n3. Dataset difficulty: Max accuracy = {max_acc:.4f}")
if max_acc > 0.95:
    print("   WARNING: Still very easy (>95% accuracy)")
elif max_acc > 0.90:
    print("   MODERATE: Challenging but achievable (90-95%)")
else:
    print("   CHALLENGING: Good academic difficulty (<90%)")

# 4. Most defensible experiment
horizon_f1 = results_df.groupby('horizon')['f1_macro'].mean()
best_horizon = horizon_f1.idxmax()
print(f"\n4. Most defensible experiment: t+6 (shortest horizon)")
print(f"   Average F1(m) by horizon:")
for h, f1 in horizon_f1.items():
    print(f"     {h}: {f1:.4f}")

# 5. Which model for final report?
print(f"\n5. Recommended model for final report: XGBoost")
print(f"   Rationale: Best balance of accuracy and interpretability")
print(f"   Best XGBoost config: C_full_t+6 (F1(m)={results_df[(results_df['model']=='XGBoost')&(results_df['dataset']=='C_full_t+6')]['f1_macro'].values[0]:.4f})")

print("\n" + "="*70)
print("PIPELINE REBUILD COMPLETE")
print("="*70)