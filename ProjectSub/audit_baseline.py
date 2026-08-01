import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

X_train = np.load('preprocessed_fixed/X_train.npy')
X_test = np.load('preprocessed_fixed/X_test.npy')
y_train = np.load('preprocessed_fixed/y_train.npy')
y_test = np.load('preprocessed_fixed/y_test.npy')

print('=== PREPROCESSED ARRAY SHAPES ===')
print(f'X_train: {X_train.shape}')
print(f'X_test: {X_test.shape}')
print(f'y_train: {y_train.shape}')
print(f'y_test: {y_test.shape}')
print()

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
print(f'Flattened X_train: {X_train_flat.shape}')
print()

# Quick RF baseline
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_flat, y_train)
pred = rf.predict(X_test_flat)
print('=== RANDOM FOREST BASELINE (quick) ===')
print(f'Accuracy: {accuracy_score(y_test, pred):.4f}')
print(f'Weighted F1: {f1_score(y_test, pred, average="weighted"):.4f}')
print()

unique, counts = np.unique(y_test, return_counts=True)
print('=== TEST CLASS DISTRIBUTION ===')
for u, c in zip(unique, counts):
    print(f'  Class {u}: {c} ({c/len(y_test)*100:.1f}%)')