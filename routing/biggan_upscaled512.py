"""BigGAN distant-shift resolution test. Identical to the original run
(prompt, pool biggan_distant_shift_1500.csv, JPEG q90, strict REAL/FAKE parser)
except each image is first resized (bicubic) so its LONGEST edge is 512 px,
real and fake alike. BigGAN fakes are 128x128 -> 512x512; reals (<=500 px) get
~512 px. Aspect ratio is preserved (no cropping/padding). 
Usage: python biggan_upscaled512.py {build|submit|collect|qwen}
"""
import base64, io, os, sys, json
import pandas as pd
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
TARGET = 512
OUTDIR = os.path.join(HERE, "results", "batch_requests")
NAME = "biggan_upscaled512"

def load_upscaled(path):
    img = Image.open(path).convert("RGB")
    s = TARGET / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    return img

def load_image_b64_upscaled(path, max_size=1024):
    img = load_upscaled(path); buf = io.BytesIO(); img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def key():
    k = os.environ.get("OPENAI_API_KEY") or open("/workspace/benchmarks_ai_research/.openai_api_key").read().strip()
    os.environ["OPENAI_API_KEY"] = k

if __name__ == "__main__":
    mode = sys.argv[1]
    pool = pd.read_csv(os.path.join(HERE, "biggan_distant_shift_1500.csv"))
    if mode in ("build", "submit"):
        from pipeline import batch_gpt55 as bg
        bg.load_image_b64 = load_image_b64_upscaled
        jl = os.path.join(OUTDIR, NAME + ".jsonl")
        if mode == "build":
            rows = [{"image_path": p, "y_true": int(y)} for p, y in zip(pool.path, pool.y_true)]
            s = bg.build_batch_request_file(rows, jl); print(s["n_rows_written"], s["jsonl_file_size_bytes"])
            # token estimate: 512px images cost ~350 in (SD1.4 GPT-5.5 calls at 512px: 339 mean);
            # output ~ mean of the existing native run
            g = pd.read_csv(os.path.join(HERE, "biggan_distant_shift_gpt55_preds.csv"))
            n = s["n_rows_written"]; mi, mo = 350, g.output_tokens.mean()
            print("est cost $%.2f (in 350, out %.0f)" % (n * (mi * 2.5 + mo * 15) / 1e6, mo))
        else:
            key(); bid = bg.submit_batch(jl, "BigGAN distant shift upscaled512"); print(bid)
            open(os.path.join(OUTDIR, NAME + ".batch_id"), "w").write(bid)
    elif mode == "collect":
        from pipeline import batch_gpt55 as bg
        key(); bid = open(os.path.join(OUTDIR, NAME + ".batch_id")).read().strip()
        st = bg.poll_batch_status(bid); print(st)
        if st["status"] == "completed":
            raw = os.path.join(OUTDIR, "outputs", NAME + ".jsonl")
            bg.download_batch_results(bid, raw)
            df = bg.parse_batch_results(raw, os.path.join(OUTDIR, NAME + ".custom_id_map.csv"))
            df["y_true"] = df["y_true"].astype(int)
            df.drop(columns=["custom_id"]).to_csv(os.path.join(HERE, "biggan_distant_shift_gpt55_upscaled512_preds.csv"), index=False)
            print(len(df), "invalid:", df.y_verifier.isna().sum(), "in/out tok", df.input_tokens.mean(), df.output_tokens.mean())
    elif mode == "qwen":
        import torch, run_qwen_sdxl_flux as rq
        processor, model, real_id, fake_id = rq.load_model()
        rows = []
        for i, (p, y) in enumerate(zip(pool.path, pool.y_true)):
            r = rq.infer_single(model, processor, real_id, fake_id, load_upscaled(p))
            rows.append({"path": p, "y_true": int(y), "y_qwen": r["y_qwen"], "raw": r["raw"],
                         "latency_qwen_ms": r["latency_qwen_ms"], "p_fake_qwen": r["p_fake_qwen"]})
            if (i + 1) % 100 == 0: print(i + 1, flush=True); torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(os.path.join(HERE, "biggan_distant_shift_qwen7b_upscaled512_preds.csv"), index=False)
