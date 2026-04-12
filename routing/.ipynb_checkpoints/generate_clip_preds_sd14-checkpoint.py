import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("/workspace/benchmarks_ai_research/routing/sd14_clip_features.csv")
clf, scaler = joblib.load("/workspace/benchmarks_ai_research/routing/sd14_clip_classifier.pkl")

X = df.drop(columns=["path", "y_true", "latency_clip_ms"]).select_dtypes(include='number').values
X = scaler.transform(X)

y_clip = clf.predict(X)
clip_proba = clf.predict_proba(X)[:, 1]

df_out = pd.DataFrame({
    "path": df["path"],
    "y_clip": y_clip,
    "clip_proba": clip_proba,
    "latency_clip_ms": df["latency_clip_ms"]
})

val_idx = np.load("/workspace/benchmarks_ai_research/routing/sd14_clip_val_idx.npy")
df_out.to_csv("/workspace/benchmarks_ai_research/routing/sd14_clip_preds.csv", index=False)

print("Saved clip_preds.csv")