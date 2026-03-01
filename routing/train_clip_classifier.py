import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("clip_features.csv")

# Keep latency column intact in CSV
# But exclude it from model features

X = df.drop(columns=["path", "y_true", "latency_clip_ms"]).values
y = df["y_true"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

pred = clf.predict(X_val)

print("CLIP classifier accuracy:", accuracy_score(y_val, pred))

joblib.dump((clf, scaler), "clip_classifier.pkl")
print("Saved clip_classifier.pkl")