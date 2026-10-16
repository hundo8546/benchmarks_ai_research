import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Paths to the CLIP features CSVs for each dataset
FEATURE_FILES = {
    "GenBuster": "/workspace/benchmarks_ai_research/routing/clip_features.csv",
    "SD14":      "/workspace/benchmarks_ai_research/routing/sd14_clip_features.csv",
    "BigGAN":    "/workspace/benchmarks_ai_research/routing/biggan_clip_features.csv",
}

def load_features(path):
    df = pd.read_csv(path)
    # Drop non-feature columns. Keep only the f0..fN embedding columns.
    drop_cols = [c for c in ["path", "y_true", "latency_clip_ms", "generator"] if c in df.columns]
    X = df.drop(columns=drop_cols).select_dtypes(include="number").values
    y = df["y_true"].values
    return X, y

# Load all three
data = {name: load_features(path) for name, path in FEATURE_FILES.items()}

results = np.zeros((3, 3))
names = list(data.keys())

for i, train_name in enumerate(names):
    X_train, y_train = data[train_name]

    # Train probe on full source dataset
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    clf = LogisticRegression(max_iter=2000, C=0.1)
    clf.fit(X_train_s, y_train)

    # Evaluate on each target dataset (including itself, for sanity)
    for j, test_name in enumerate(names):
        X_test, y_test = data[test_name]
        X_test_s = scaler.transform(X_test)
        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        results[i, j] = acc
        print(f"Train={train_name:10s}  Test={test_name:10s}  Acc={acc:.3f}")

# Pretty print as a transfer matrix
print("\nTransfer accuracy matrix (rows=train, cols=test):")
print(f"{'':12s}", end="")
for n in names:
    print(f"{n:>12s}", end="")
print()
for i, n in enumerate(names):
    print(f"{n:12s}", end="")
    for j in range(3):
        marker = "*" if i == j else " "
        print(f"{results[i,j]:>11.3f}{marker}", end="")
    print()
print("* = in-domain (diagonal)")

# Save to file
pd.DataFrame(results, index=names, columns=names).to_csv(
    "/workspace/benchmarks_ai_research/routing/clip_transfer_matrix.csv"
)