import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

df = pd.read_csv("/workspace/benchmarks_ai_research/routing/sd14_clip_features.csv")

# Check class distribution
print("Class distribution in full dataset:")
print(df["y_true"].value_counts())

# Features
X = df.drop(columns=["path", "y_true", "latency_clip_ms", "generator"]).values
y = df["y_true"].values
groups = df["generator"].values
print("\nGenerator distribution:")
print(df["generator"].value_counts())

# First split: separate clip vs bandit data
# Use stratified split to maintain class distribution
clip_idx, bandit_idx = train_test_split(
    np.arange(len(X)), 
    test_size=0.5, 
    random_state=42, 
    stratify=y  # This ensures both splits have similar class distribution
)

# Check class distribution in clip portion
y_clip_initial = y[clip_idx]
print(f"\nClass distribution in clip portion (before further split):")
print(f"Class 0: {np.sum(y_clip_initial == 0)}")
print(f"Class 1: {np.sum(y_clip_initial == 1)}")

# Then group split within clip portion for CLIP train/val
X_clip = X[clip_idx]
y_clip = y[clip_idx]
groups_clip = groups[clip_idx]

# Use GroupShuffleSplit but ensure both classes are present
gss = GroupShuffleSplit(test_size=0.2, random_state=42)

# Try multiple splits until we find one with both classes in training
max_attempts = 10
for attempt in range(max_attempts):
    train_rel, val_rel = next(gss.split(X_clip, y_clip, groups_clip))
    train_idx = clip_idx[train_rel]
    val_idx = clip_idx[val_rel]
    
    y_train_check = y[train_idx]
    y_val_check = y[val_idx]
    
    # Check if both classes are present in training set
    if len(np.unique(y_train_check)) >= 2:
        print(f"\nFound valid split on attempt {attempt + 1}")
        print(f"Training set classes: {np.unique(y_train_check)}")
        print(f"Validation set classes: {np.unique(y_val_check)}")
        break
    else:
        print(f"Attempt {attempt + 1}: Training set has only class {np.unique(y_train_check)}")
        # Regenerate the GroupShuffleSplit iterator
        gss = GroupShuffleSplit(test_size=0.2, random_state=42 + attempt)
else:
    # If no split found, fall back to simple stratified split
    print("\nNo valid GroupShuffleSplit found. Using simple stratified split...")
    train_idx, val_idx = train_test_split(
        clip_idx, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_clip
    )

# Save validation indices
np.save("/workspace/benchmarks_ai_research/routing/sd14_clip_val_idx.npy", val_idx)

X_train = X[train_idx]
X_val = X[val_idx]
y_train = y[train_idx]
y_val = y[val_idx]

# Verify final class distribution
print(f"\nFinal training set size: {len(y_train)}")
print(f"Final validation set size: {len(y_val)}")
print(f"Training class distribution - Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
print(f"Validation class distribution - Class 0: {np.sum(y_val == 0)}, Class 1: {np.sum(y_val == 1)}")

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Regularized logistic regression with class weights
clf = LogisticRegression(
    max_iter=2000,
    C=0.1,
    class_weight=class_weight_dict,  # Handle class imbalance
    random_state=42
)

clf.fit(X_train, y_train)

pred = clf.predict(X_val)
print(f"\nCLIP classifier accuracy: {accuracy_score(y_val, pred):.4f}")

# Also print classification report for more details
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_val, pred))

# Save model
joblib.dump((clf, scaler), "/workspace/benchmarks_ai_research/routing/sd14_clip_classifier.pkl")
print("\nSaved clip_classifier.pkl")