"""
BigGAN distant-shift experiment (reconstructing the paper's biggan_shift
table): GPT-5.5 on the 1,500-sample stratified disagreement subset where
SD1.4-trained stale FFT+SigLIP2 probes disagree on GenImage BigGAN.
"""
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))
import e8_modern_diffusion as e8
from fix_gpt55_blanks import load_image_b64_any

ROUTING_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = "gpt-5.5"
OUT_FILE = os.path.join(ROUTING_BASE, "biggan_distant_shift_gpt55_preds.csv")


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        with open("/root/.openai_api_key") as f:
            api_key = f.read().strip()
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    df = pd.read_csv(os.path.join(ROUTING_BASE, "biggan_distant_shift_1500.csv"))
    done_rows, done_paths = [], set()
    if os.path.exists(OUT_FILE):
        df_old = pd.read_csv(OUT_FILE)
        done_rows = df_old.to_dict("records")
        done_paths = set(df_old["path"])

    todo = df[~df["path"].isin(done_paths)].reset_index(drop=True)
    print(f"total={len(df)} already_done={len(done_paths)} todo={len(todo)}")

    for i, r in enumerate(tqdm(todo.itertuples(), total=len(todo))):
        try:
            b64 = load_image_b64_any(r.path)
            y, raw, in_tok, out_tok, lat = e8.call_gpt55(client, MODEL, b64)
        except Exception as ex:
            y, raw, in_tok, out_tok, lat = None, f"ERROR:{ex}", 0, 0, 0
        done_rows.append({
            "path": r.path, "y_true": int(r.y_true), "y_verifier": y, "raw": raw,
            "latency_ms": lat, "input_tokens": in_tok, "output_tokens": out_tok,
        })
        if (i + 1) % 25 == 0:
            pd.DataFrame(done_rows).to_csv(OUT_FILE, index=False)
            fake_ct = sum(1 for x in done_rows if str(x["raw"]).upper().find("FAKE") >= 0 and str(x["raw"]).upper().find("REAL") < 0)
            real_ct = sum(1 for x in done_rows if str(x["raw"]).upper().find("REAL") >= 0 and str(x["raw"]).upper().find("FAKE") < 0)
            print(f"  [checkpoint {i+1}/{len(todo)}] label so far: REAL={real_ct} FAKE={fake_ct}")

    pd.DataFrame(done_rows).to_csv(OUT_FILE, index=False)
    print(f"Saved {OUT_FILE} ({len(done_rows)} rows)")


if __name__ == "__main__":
    main()
