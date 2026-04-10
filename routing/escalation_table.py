from analysis2 import res_gb, df_gb
import numpy as np
bandit_esc = np.array(res_gb["Bandit"]["escalated"])
disagree_esc = np.array(res_gb["Disagree thresh"]["escalated"])
cnn_correct = df_gb["cnn_correct"].values
disagree = df_gb["disagree"].values
for case_name, mask in [
    ("CNN correct, agree", (cnn_correct==1) & (disagree==0)),
    ("CNN correct, disagree", (cnn_correct==1) & (disagree==1)),
    ("CNN wrong, agree", (cnn_correct==0) & (disagree==0)),
    ("CNN wrong, disagree", (cnn_correct==0) & (disagree==1)),
]:
    print(f"{case_name}: n={mask.sum()}, bandit={bandit_esc[mask].mean():.2f}, disagree={disagree_esc[mask].mean():.2f}")

# In analysis2 context
for gen in sorted(df_gb["generator"].unique()):
    mask = df_gb["generator"] == gen
    n = mask.sum()
    cnn_acc = df_gb.loc[mask, "cnn_correct"].mean()
    clip_acc = df_gb.loc[mask, "clip_correct"].mean()
    dis_pred = np.where(df_gb.loc[mask, "disagree"]==1, df_gb.loc[mask, "y_qwen"], df_gb.loc[mask, "y_cnn"])
    dis_acc = (dis_pred == df_gb.loc[mask, "y_true"]).mean()
    ens_pred = df_gb.loc[mask, "y_clip"]
    ens_acc = (ens_pred == df_gb.loc[mask, "y_true"]).mean()
    print(f"{gen} & {n} & {cnn_acc:.3f} & ... & {dis_acc:.3f} & {ens_acc:.3f} & {clip_acc:.3f}")