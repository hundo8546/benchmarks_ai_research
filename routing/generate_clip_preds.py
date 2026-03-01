import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("clip_features.csv")

clf, scaler = joblib.load("clip_classifier.pkl")

X = df.drop(columns=["path", "y_true","latency_clip_ms"]).values
X = scaler.transform(X)

y_clip = clf.predict(X)

df_out = pd.DataFrame({
    "path": df["path"],
    "y_clip": y_clip,
    "latency_clip_ms":df["latency_clip_ms"]
})

df_out.to_csv("clip_preds.csv", index=False)

print("Saved clip_preds.csv")