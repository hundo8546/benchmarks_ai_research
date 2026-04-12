import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

df = pd.read_csv("/workspace/benchmarks_ai_research/routing/clip_features.csv")

# Features
X = df.drop(columns=["path", "y_true", "latency_clip_ms", "generator"]).values
y = df["y_true"].values
groups = df["generator"].values
print(df["generator"].value_counts())
# Split by generator (prevents leakage)
clip_idx, bandit_idx = train_test_split(
    np.arange(len(X)), test_size=0.5, random_state=42, stratify=y
)

# Then group split within clip portion for CLIP train/val
X_clip = X[clip_idx]
y_clip = y[clip_idx]
groups_clip = groups[clip_idx]

gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_rel, val_rel = next(gss.split(X_clip, y_clip, groups_clip))
train_idx = clip_idx[train_rel]
val_idx = clip_idx[val_rel]

np.save("/workspace/benchmarks_ai_research/routing/clip_val_idx.npy", bandit_idx)

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