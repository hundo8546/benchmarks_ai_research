import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("/workspace/benchmarks_ai_research/routing/clip_features.csv")

# Features
X = df.drop(columns=["path", "y_true", "latency_clip_ms", "generator"]).values
y = df["y_true"].values
groups = df["generator"].values
print(df["generator"].value_counts())
# Split by generator (prevents leakage)
gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups))
np.save("/workspace/benchmarks_ai_research/routing/clip_val_idx.npy", val_idx)

X_train = X[train_idx]
X_val = X[val_idx]

y_train = y[train_idx]
y_val = y[val_idx]

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Regularized logistic regression
clf = LogisticRegression(
    max_iter=2000,
    C=0.1
)

clf.fit(X_train, y_train)

pred = clf.predict(X_val)

print("CLIP classifier accuracy:", accuracy_score(y_val, pred))

# Save model
joblib.dump((clf, scaler), "clip_classifier.pkl")
print("Saved clip_classifier.pkl")