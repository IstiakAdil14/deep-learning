import pandas as pd
import numpy as np

df = pd.read_csv('smart_grid_dataset_sylhet.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Per-upazila split analysis (replicate notebook logic)
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

print('=== PER-UPZILA SPLIT ===')
print(f'Train rows: {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)')
print(f'Test rows: {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)')
print()

# Check for timestamp overlap between train and test
train_times = set(df_train['datetime'])
test_times = set(df_test['datetime'])
overlap = train_times & test_times
print(f'=== TEMPORAL OVERLAP CHECK ===')
print(f'Unique timestamps in train: {len(train_times)}')
print(f'Unique timestamps in test: {len(test_times)}')
print(f'Timestamps in BOTH train and test: {len(overlap)}')
if len(overlap) > 0:
    print(f'Overlap sample: {list(overlap)[:5]}')
print()

# Check boundary conditions per upazila
print('=== PER-UPZILA BOUNDARY CHECK ===')
boundary_issues = 0
for upazila_name, group in df.groupby("upazila", sort=False):
    group = group.sort_values("datetime").reset_index(drop=True)
    split_idx = int(len(group) * TRAIN_RATIO)
    if split_idx > 0 and split_idx < len(group):
        train_last = group.iloc[split_idx-1]['datetime']
        test_first = group.iloc[split_idx]['datetime']
        if train_last == test_first:
            boundary_issues += 1
            if boundary_issues <= 3:
                print(f'  {upazila_name}: train_last={train_last}, test_first={test_first}')

print(f'Total upazilas with boundary timestamp overlap: {boundary_issues}')
print()

# Target leakage analysis
print('=== TARGET LEAKAGE: demand_index ===')
print('demand_index thresholds by risk level:')
for risk in ['Low', 'Medium', 'High']:
    subset = df[df['risk_level'] == risk]['demand_index']
    print(f'  {risk}: min={subset.min():.4f}, max={subset.max():.4f}, mean={subset.mean():.4f}')

print()
print('Class separation by demand_index:')
print(f'Max Low demand_index: {df[df["risk_level"]=="Low"]["demand_index"].max():.4f}')
print(f'Min High demand_index: {df[df["risk_level"]=="High"]["demand_index"].min():.4f}')
separability = 1.0 - (df[df["risk_level"]=="Low"]["demand_index"].max() - df[df["risk_level"]=="High"]["demand_index"].min())
print(f'Gap between classes: {df[df["risk_level"]=="High"]["demand_index"].min() - df[df["risk_level"]=="Low"]["demand_index"].max():.4f}')
print()

# Correlation analysis
print('=== CORRELATION MATRIX ===')
corr = df[['temperature','humidity','rainfall','demand_index']].corr()
print(corr.round(4))