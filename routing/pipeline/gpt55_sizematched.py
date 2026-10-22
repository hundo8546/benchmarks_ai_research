"""
routing/pipeline/gpt55_sizematched.py

Step 3 of the SDXL/FLUX resolution-confound control (remainingupdates1.md):
GPT-5.5 on the size-matched copies (built by
pipeline/robustness/build_sizematched_sdxl_flux.py) of every image in
results/batch_requests/CONSOLIDATED_full_sdxl_flux.csv.

Everything except the image path is identical to the original uniform run:
the request file is built by batch_gpt55.build_batch_request_file (same
VERIFIER_PROMPT, same load_image_b64 -- max edge 1024 px, JPEG q90 -- same
model id, same 4096 max_output_tokens, same /v1/responses endpoint, same 24h
completion window) and parsed by batch_gpt55.parse_batch_results (same strict
_is_valid_answer parser). custom_ids are sha1(size-matched path), so they can
never collide with the original run's ids.

Subcommands (each network step needs OPENAI_API_KEY; batch_gpt55 enforces it):
  build  --set smoke|full     write results/batch_requests/sizematched_<set>.jsonl
                              (+ .custom_id_map.csv); no network
  submit --set smoke|full     upload + submit (auto-splits files > 195 MB);
                              batch ids appended to sizematched_<set>.batch_ids
  poll   --set smoke|full     print status of every submitted batch
  mopup  --set smoke|full     Standard-tier synchronous retry (verifiers.call_gpt55) of every
                              request without a valid batch answer (cancelled/expired/invalid);
                              writes sizematched_<set>_mopup_standard.csv
  parse  --set smoke|full     download + parse results (+ mop-up rows); writes
                              sizematched_<set>_results.csv and, for --set full,
                              CONSOLIDATED_full_sdxl_flux_sizematched.csv
  tokens --set smoke|full     input_tokens distribution by class (parity check)

The smoke set is 20 images: 5 SDXL fakes, 5 FLUX fakes, 10 reals, drawn with
seed 42 from the CONSOLIDATED set.
"""
import argparse
import os
import sys

import pandas as pd

ROUTING_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROUTING_BASE)
from pipeline import batch_gpt55 as bg
from pipeline.robustness.build_sizematched_sdxl_flux import sm_path_for

BATCH_DIR = os.path.join(ROUTING_BASE, "results", "batch_requests")
UNIFORM_CSV = os.path.join(BATCH_DIR, "CONSOLIDATED_full_sdxl_flux.csv")
OUT_CONSOLIDATED = os.path.join(BATCH_DIR, "CONSOLIDATED_full_sdxl_flux_sizematched.csv")


def _paths(set_name):
    stem = os.path.join(BATCH_DIR, f"sizematched_{set_name}")
    return {"jsonl": stem + ".jsonl", "map": stem + ".custom_id_map.csv",
            "ids": stem + ".batch_ids", "raw_dir": stem + "_outputs",
            "mopup": stem + "_mopup_standard.csv", "results": stem + "_results.csv"}


def source_rows(set_name):
    df = pd.read_csv(UNIFORM_CSV)[["path", "y_true"]].rename(columns={"path": "orig_path"})
    if set_name == "smoke":
        fake = df[df.y_true == 1]
        sdxl = fake[fake.orig_path.str.contains("/sdxl/")].sample(5, random_state=42)
        flux = fake[~fake.orig_path.str.contains("/sdxl/")].sample(5, random_state=42)
        real = df[df.y_true == 0].sample(10, random_state=42)
        df = pd.concat([sdxl, flux, real], ignore_index=True)
    df["sm_path"] = df["orig_path"].map(sm_path_for)
    missing = ~df["sm_path"].map(os.path.exists)
    if missing.any():
        raise FileNotFoundError(f"{int(missing.sum())} size-matched images missing, e.g. "
                                f"{df.loc[missing, 'sm_path'].iloc[0]}; run build_sizematched_sdxl_flux.py")
    return df


def cmd_build(set_name):
    df = source_rows(set_name)
    p = _paths(set_name)
    rows = [{"custom_id": bg._stable_custom_id(r.sm_path), "image_path": r.sm_path, "y_true": int(r.y_true)}
            for r in df.itertuples(index=False)]
    summary = bg.build_batch_request_file(rows, p["jsonl"])
    if summary["n_rows_skipped"]:
        raise RuntimeError(f"{summary['n_rows_skipped']} rows skipped: {summary['skipped'][:3]}")
    print(f"{set_name}: {summary['n_rows_written']} requests "
          f"(real={int((df.y_true == 0).sum())}, fake={int((df.y_true == 1).sum())})")


def cmd_submit(set_name):
    p = _paths(set_name)
    if os.path.exists(p["ids"]):
        raise RuntimeError(f"{p['ids']} exists -- batches already submitted for '{set_name}'; refusing to resubmit")
    parts = bg.split_batch_request_file(p["jsonl"])
    ids = []
    for i, part in enumerate(parts, start=1):
        desc = f"sizematched_{set_name} SDXL/FLUX 512px resolution control" + (f" (part {i})" if len(parts) > 1 else "")
        ids.append(bg.submit_batch(part, description=desc))
    with open(p["ids"], "w") as f:
        f.write("\n".join(ids) + "\n")
    print(f"submitted {len(ids)} batch(es) -> {p['ids']}")


def _batch_ids(set_name):
    with open(_paths(set_name)["ids"]) as f:
        return [l.strip() for l in f if l.strip()]


def cmd_poll(set_name):
    for bid in _batch_ids(set_name):
        s = bg.poll_batch_status(bid)
        print(f"{bid}: {s['status']}  completed={s['request_counts_completed']} "
              f"failed={s['request_counts_failed']} total={s['request_counts_total']}")


def _parse_batches(set_name):
    """Valid-or-not answers from every batch part. A cancelled or expired part
    still returns its completed requests; its unfinished ones are simply absent."""
    p = _paths(set_name)
    os.makedirs(p["raw_dir"], exist_ok=True)
    frames = []
    for bid in _batch_ids(set_name):
        out = os.path.join(p["raw_dir"], f"{bid}.jsonl")
        if not os.path.exists(out):
            s = bg.poll_batch_status(bid)
            if s["status"] not in ("completed", "cancelled", "expired"):
                raise RuntimeError(f"{bid} is still {s['status']}; wait, or cancel it first")
            bg.download_batch_results(bid, out)
        frames.append(bg.parse_batch_results(out, custom_id_map_path=p["map"]))
    res = pd.concat(frames, ignore_index=True)
    res["tier"], res["attempts_used"] = "batch", 1
    return res


def _stragglers(set_name, res):
    """custom_id map rows with no valid batch answer: unfinished (cancelled/expired) or invalid."""
    idmap = pd.read_csv(_paths(set_name)["map"], dtype={"custom_id": str})
    ok = set(res.loc[res["y_verifier"].notna(), "custom_id"])
    return idmap[~idmap["custom_id"].isin(ok)]


def cmd_mopup(set_name):
    """Standard-tier synchronous retry of the stragglers via verifiers.call_gpt55 (same model,
    prompt and load_image_b64; escalating [2048, 4096] budget on empty/invalid answers), the
    procedure batch_gpt55's module docstring prescribes. Rows already in the mop-up file are
    skipped, so a rerun never re-bills them."""
    from openai import OpenAI
    from pipeline import verifiers as vf
    p = _paths(set_name)
    todo = _stragglers(set_name, _parse_batches(set_name))
    done = pd.read_csv(p["mopup"], dtype={"custom_id": str}) if os.path.exists(p["mopup"]) else None
    if done is not None:
        todo = todo[~todo["custom_id"].isin(done["custom_id"])]
    print(f"{len(todo)} stragglers to run synchronously")
    client = OpenAI(api_key=bg._require_api_key())
    rows = [] if done is None else done.to_dict("records")
    for r in todo.itertuples(index=False):
        g = vf.call_gpt55(client, bg.GPT55_MODEL_ID, vf.load_image_b64(r.image_path))
        rows.append({"path": r.image_path, "y_true": r.y_true, "y_verifier": g.y, "raw": g.raw,
                     "latency_ms": g.latency_ms, "input_tokens": g.input_tokens, "output_tokens": g.output_tokens,
                     "custom_id": r.custom_id, "tier": "standard", "attempts_used": g.attempts_used})
        pd.DataFrame(rows).to_csv(p["mopup"], index=False)
        print(f"  {r.custom_id[:8]} y_true={r.y_true} -> {g.raw!r} (attempts={g.attempts_used}, "
              f"in={g.input_tokens}, out={g.output_tokens})")


def cmd_parse(set_name):
    p = _paths(set_name)
    res = _parse_batches(set_name)
    if os.path.exists(p["mopup"]):
        mop = pd.read_csv(p["mopup"], dtype={"custom_id": str})
        res = pd.concat([res[~res["custom_id"].isin(mop["custom_id"])], mop], ignore_index=True)
    src = source_rows(set_name)[["orig_path", "sm_path"]]
    res = res.merge(src, left_on="path", right_on="sm_path", how="left").drop(columns="sm_path")
    res.to_csv(p["results"], index=False)
    missing = _stragglers(set_name, res)
    print(f"parsed {len(res)} rows -> {p['results']}; {len(missing)} of "
          f"{len(missing) + int(res['y_verifier'].notna().sum())} requests lack a valid REAL/FAKE answer; "
          f"tiers: {res['tier'].value_counts().to_dict()}")
    if set_name == "full" and len(missing) == 0:
        res[["path", "orig_path", "y_true", "y_verifier", "raw", "input_tokens", "output_tokens",
             "custom_id", "tier"]].to_csv(OUT_CONSOLIDATED, index=False)
        print(f"wrote {OUT_CONSOLIDATED}")
    elif len(missing):
        print("run `mopup` before writing CONSOLIDATED")


def cmd_tokens(set_name):
    res = pd.read_csv(_paths(set_name)["results"])
    print(res.groupby("y_true")["input_tokens"].describe())
    print("distinct input_tokens per class:",
          {int(k): sorted(v.unique().tolist())[:10] for k, v in res.groupby("y_true")["input_tokens"]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "submit", "poll", "mopup", "parse", "tokens"])
    ap.add_argument("--set", choices=["smoke", "full"], required=True)
    a = ap.parse_args()
    {"build": cmd_build, "submit": cmd_submit, "poll": cmd_poll, "mopup": cmd_mopup,
     "parse": cmd_parse, "tokens": cmd_tokens}[a.cmd](a.set)
