"""
Run GPT-5.5 on the NEW disagreement-subset samples introduced by the
expanded (>=2000-sample) SD14/BigGAN val splits that don't already have a
GPT-5.5 prediction from the old (400/480-sample) run. Appends to the
existing {prefix}gpt55_disagree_preds.csv files.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "extractors"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))

from extractors.common import DATASETS, load_val_paths
import e8_modern_diffusion as e8

ROUTING_BASE = os.path.dirname(os.path.abspath(__file__))
DATASETS_TO_RUN = ["SD14", "BigGAN"]
MODEL = "gpt-5.5"


def get_disagree_subset(ds_name):
    cfg = DATASETS[ds_name]
    prefix = cfg["prefix"]
    df_fft = pd.read_csv(os.path.join(ROUTING_BASE, f"{prefix}fft_features.csv"))
    fft_cols = [c for c in df_fft.columns if c.startswith("f")]
    clf_fft, sc_fft = joblib.load(os.path.join(ROUTING_BASE, f"{prefix}fft_probe.pkl"))
    df_sig = pd.read_csv(os.path.join(ROUTING_BASE, f"{prefix}siglip2_features.csv"))
    sig_cols = [c for c in df_sig.columns if c.startswith("f")]
    clf_sig, sc_sig = joblib.load(os.path.join(ROUTING_BASE, f"{prefix}siglip2_probe.pkl"))
    val_paths = load_val_paths(cfg, df_fft["path"].tolist())

    df_fft_v = df_fft[df_fft["path"].isin(val_paths)].copy()
    df_fft_v["y_fft"] = clf_fft.predict(sc_fft.transform(df_fft_v[fft_cols].values.astype(np.float32)))
    df_sig_v = df_sig[df_sig["path"].isin(val_paths)].copy()
    df_sig_v["y_sig"] = clf_sig.predict(sc_sig.transform(df_sig_v[sig_cols].values.astype(np.float32)))
    df = df_fft_v[["path", "y_true", "y_fft"]].merge(df_sig_v[["path", "y_sig"]], on="path")
    df["disagree"] = df["y_fft"] != df["y_sig"]
    return df[df["disagree"]].reset_index(drop=True)


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        with open("/root/.openai_api_key") as f:
            api_key = f.read().strip()

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    for ds_name in DATASETS_TO_RUN:
        cfg = DATASETS[ds_name]
        prefix = cfg["prefix"]
        gpt_file = os.path.join(ROUTING_BASE, f"{prefix}gpt55_disagree_preds.csv")

        dis = get_disagree_subset(ds_name)
        done_rows = []
        done_paths = set()
        if os.path.exists(gpt_file):
            df_old = pd.read_csv(gpt_file)
            done_rows = df_old.to_dict("records")
            done_paths = set(df_old["path"])

        todo = dis[~dis["path"].isin(done_paths)].reset_index(drop=True)
        print(f"\n=== {ds_name} === disagree={len(dis)}  already_have={len(done_paths & set(dis['path']))}  todo={len(todo)}")

        for i, r in enumerate(tqdm(todo.itertuples(), total=len(todo), desc=ds_name)):
            try:
                b64 = e8.load_image_b64(r.path)
                y, raw, in_tok, out_tok, lat = e8.call_gpt55(client, MODEL, b64)
            except Exception as ex:
                y, raw, in_tok, out_tok, lat = 1, f"ERROR:{ex}", 0, 0, 0
            done_rows.append({
                "path": r.path, "y_true": int(r.y_true),
                "y_verifier": y, "raw": raw,
                "latency_ms": lat, "input_tokens": in_tok, "output_tokens": out_tok,
            })
            if (i + 1) % 25 == 0:
                pd.DataFrame(done_rows).to_csv(gpt_file, index=False)

        pd.DataFrame(done_rows).to_csv(gpt_file, index=False)
        print(f"Saved {gpt_file} ({len(done_rows)} total rows)")


if __name__ == "__main__":
    main()
