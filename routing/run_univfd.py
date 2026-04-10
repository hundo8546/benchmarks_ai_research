import sys
sys.path.insert(0, '/workspace/benchmarks_ai_research/UniversalFakeDetect')

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from models.clip_models import CLIPModel

BASE = "/workspace/benchmarks_ai_research/routing/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel("ViT-L/14")
weights = torch.load(
    "/workspace/benchmarks_ai_research/UniversalFakeDetect/pretrained_weights/fc_weights.pth",
    map_location=device
)
model.fc.load_state_dict(weights)
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                         [0.26862954, 0.26130258, 0.27577711])
])

def run_inference(paths, y_true, label):
    preds, probs = [], []
    for i, path in enumerate(paths):
        if i % 50 == 0:
            print(f"{label}: {i}/{len(paths)}")
        try:
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(x)
                prob = torch.sigmoid(logit).item()
            pred = 1 if prob > 0.5 else 0
            preds.append(pred)
            probs.append(prob)
        except Exception as e:
            print(f"Error on {path}: {e}")
            preds.append(0)
            probs.append(0.0)

    preds = np.array(preds)
    y_true = np.array(y_true)
    acc = np.mean(preds == y_true)
    fake_recall = np.mean(preds[y_true==1] == 1) if (y_true==1).sum() > 0 else 0
    real_recall = np.mean(preds[y_true==0] == 0) if (y_true==0).sum() > 0 else 0
    bal_acc = (fake_recall + real_recall) / 2
    print(f"\n{label} UnivFD results:")
    print(f"  Accuracy:            {acc:.3f}")
    print(f"  Balanced acc:        {bal_acc:.3f}")
    print(f"  Fake recall:         {fake_recall:.3f}")
    print(f"  Real recall:         {real_recall:.3f}")
    print(f"  Predicted fake rate: {preds.mean():.3f}")
    return preds, probs, acc, bal_acc

for prefix, vp_file, label in [
    ("", "bandit_val_paths.npy", "GenBuster"),
    ("sd14_", "sd14_bandit_val_paths.npy", "SD14"),
    ("biggan_", "biggan_bandit_val_paths.npy", "BigGAN"),
]:
    df = pd.read_csv(BASE + f"{prefix}bandit_dataset.csv")
    val_paths = np.load(BASE + vp_file, allow_pickle=True)
    df = df[df["path"].isin(val_paths)].reset_index(drop=True)
    preds, probs, acc, bal_acc = run_inference(
        df["path"].tolist(), df["y_true"].tolist(), label
    )
    df["y_univfd"] = preds
    df["univfd_prob"] = probs
    df[["path", "y_true", "y_univfd", "univfd_prob"]].to_csv(
        BASE + f"{prefix}univfd_preds.csv", index=False
    )
    print(f"Saved {prefix}univfd_preds.csv")
