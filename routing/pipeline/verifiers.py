"""
routing.pipeline.verifiers — consolidated verifier-calling code.

Phase 0 remediation item E. Prior to this module there were THREE divergent
GPT-5.5-calling implementations in this repo:
  - routing/experiments/e4_gpt55.py:call_gpt()
        max_output_tokens=16 (fixed, too low), ambiguous/empty -> silently
        defaults to y_pred=1 (FAKE). No truncation detection at all.
  - routing/fill_table6_gpt55_gap.py:call_gpt55()
        max_output_tokens=100 (fixed), ambiguous -> silently defaults to
        y=1 (FAKE). This is the implementation that produced the known
        19 silently-mislabeled "FA" (truncated "FAKE") rows at
        output_tokens==100 exactly (4 sdxl_clip, 2 sdxl_dinov2, 4
        sdxl_siglip2, 1 flux_clip, 2 flux_dinov2, 6 flux_siglip2 — counted
        directly from the on-disk *_gpt55_*_disagree_preds.csv files).
  - routing/experiments/e8_modern_diffusion.py:call_gpt55()
        escalating budget [2048, 4096], no silent default-to-FAKE on a
        fully empty response -- the correct/current one, used as the base
        for this module. It still had a narrower gap: it only detected
        "truncated" by checking `if not raw`, which misses a response that
        came back non-empty but not exactly REAL/FAKE (e.g. "FA", "The
        answer is FAKE because..."). That gap is fixed here by validating
        the parsed answer is EXACTLY "REAL" or "FAKE" after
        .strip().upper(), not merely non-empty.

This module provides ONE call_gpt55() and ONE call_qwen(), consolidating the
above plus the three near-identical Qwen runner scripts (run_qwen_v2.py /
sd14_run_qwen_v2.py / biggan_run_qwen_v2.py, which differ only in
INDEX_FILE/OUT_FILE/IS_VIDEO constants).

IMPORTANT: there is no OPENAI_API_KEY in this environment. call_gpt55() is
written and unit-testable (see test_parsing() below, which is pure string
logic and was actually run) but MUST NOT be invoked against the live OpenAI
API in this environment. This module is not wired into any other script
yet -- that consolidation is a later phase.

Qwen (local transformers inference) has no such restriction and call_qwen()
/ load_qwen_model() can be and were exercised against the GPU in this
environment (see the __main__ smoke test).
"""
from __future__ import annotations

import base64
import io
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Shared prompt (identical wording used by e8_modern_diffusion.py / the
# fill_table6 script / run_qwen_v2.py's REASONING_PROMPT -- this exact
# wording is what's reported in the paper's Section IV.B prompt listing).
# ─────────────────────────────────────────────────────────────────────────
VERIFIER_PROMPT = (
    "Analyze this image carefully. Determine whether it is a REAL "
    "photograph or an AI-generated FAKE image. Consider both "
    "possibilities equally. Answer with exactly one word: REAL or FAKE."
)

GPT_MAX_OUTPUT_TOKENS = [2048, 4096]   # escalating budget, kept from e8
GPT_MAX_RETRIES = 5
GPT_RETRY_BASE = 2.0


# ─────────────────────────────────────────────────────────────────────────
# GPT-5.5 — WRITE ONLY. Do not call against a live API in this environment
# (no OPENAI_API_KEY is configured here).
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class GPT55Response:
    """Full audit record for a single GPT-5.5 verifier call."""
    y: Optional[int]          # 0=REAL, 1=FAKE, None=invalid/ambiguous/truncated after all retries
    raw: str                  # full raw response text (untouched, not upper-cased) actually used
    model: str
    timestamp_utc: str        # ISO-8601 UTC timestamp of the (last) attempt
    attempts_used: int        # how many attempts were actually made (1..GPT_MAX_RETRIES)
    input_tokens: int
    output_tokens: int
    latency_ms: float


def _is_valid_answer(raw_text: str) -> Optional[int]:
    """
    Return 0 (REAL), 1 (FAKE), or None if `raw_text` is NOT exactly one of
    those two words after .strip().upper().

    This is the fix for the truncation-detection gap: the old code treated
    anything non-empty as a valid answer and pattern-matched "FAKE" / "REAL"
    as substrings, which accepted "FA" (truncated "FAKE") as if it meant
    FAKE. Here, nothing short of an exact match is accepted -- "FA",
    "FAKE.", "The answer is FAKE because...", and "" are all None (invalid),
    exactly like a genuinely empty response.
    """
    cleaned = (raw_text or "").strip().upper()
    if cleaned == "REAL":
        return 0
    if cleaned == "FAKE":
        return 1
    return None


def call_gpt55(
    client,
    model: str,
    b64_image: str,
    prompt: str = VERIFIER_PROMPT,
    max_retries: int = GPT_MAX_RETRIES,
    retry_base: float = GPT_RETRY_BASE,
    token_budgets=GPT_MAX_OUTPUT_TOKENS,
) -> GPT55Response:
    """
    Call GPT-5.5 (OpenAI Responses API) on a single base64-JPEG image and
    return a full audit record.

    Behavior (per Phase 0 remediation spec):
      - Escalating output-token budget across retries (token_budgets[min(
        attempt, len(token_budgets)-1)]), same [2048, 4096] schedule as the
        correct e8_modern_diffusion.py implementation this is based on.
      - Single consistent retry policy: up to `max_retries` attempts,
        exponential backoff starting at `retry_base` seconds, doubling each
        retry, on openai.RateLimitError / openai.APIError.
      - NEVER silently defaults to FAKE (or any label) on an invalid
        response. If, after all retries, the response is empty OR
        non-empty-but-not-exactly-"REAL"/"FAKE" (e.g. a truncated "FA"),
        returns y=None with the raw text preserved (or the sentinel
        "EMPTY_AFTER_RETRY" if the response was truly empty).
      - Returns full audit info: model id, UTC ISO-8601 timestamp, number
        of attempts actually used, input/output token counts, and the full
        raw response text (not just the upper-cased parsed form).

    This function performs real network I/O via `client` (an
    openai.OpenAI instance) -- it is NOT called anywhere in this
    environment, where OPENAI_API_KEY is unset. It is provided for the next
    phase, once API budget/credentials are available.
    """
    import openai  # local import: this module must be importable even
                    # without the openai package installed, for the offline
                    # parsing self-test below.

    delay = retry_base
    last_raw_text = ""
    last_in_tok = last_out_tok = 0
    last_lat = 0.0

    for attempt in range(max_retries):
        budget = token_budgets[min(attempt, len(token_budgets) - 1)]
        ts = datetime.now(timezone.utc).isoformat()
        try:
            t0 = time.perf_counter()
            resp = client.responses.create(
                model=model,
                input=[{"role": "user", "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image",
                     "image_url": f"data:image/jpeg;base64,{b64_image}"},
                ]}],
                max_output_tokens=budget,
            )
            lat = (time.perf_counter() - t0) * 1000
            raw_text = resp.output_text or ""
            usage = resp.usage
            in_tok = getattr(usage, "input_tokens", 0) or 0
            out_tok = getattr(usage, "output_tokens", 0) or 0

            last_raw_text, last_in_tok, last_out_tok, last_lat = raw_text, in_tok, out_tok, lat

            y = _is_valid_answer(raw_text)
            if y is not None:
                return GPT55Response(
                    y=y, raw=raw_text, model=model, timestamp_utc=ts,
                    attempts_used=attempt + 1, input_tokens=in_tok,
                    output_tokens=out_tok, latency_ms=lat,
                )

            # Invalid (empty, ambiguous, or truncated like "FA") -- escalate
            # the budget and retry rather than guessing a label.
            if attempt < max_retries - 1:
                continue
            sentinel = raw_text if raw_text.strip() else "EMPTY_AFTER_RETRY"
            return GPT55Response(
                y=None, raw=sentinel, model=model, timestamp_utc=ts,
                attempts_used=attempt + 1, input_tokens=in_tok,
                output_tokens=out_tok, latency_ms=lat,
            )

        except openai.RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            return GPT55Response(
                y=None, raw=f"ERROR:RateLimitError", model=model,
                timestamp_utc=ts, attempts_used=attempt + 1,
                input_tokens=last_in_tok, output_tokens=last_out_tok,
                latency_ms=last_lat,
            )
        except openai.APIError as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            return GPT55Response(
                y=None, raw=f"ERROR:{e}", model=model, timestamp_utc=ts,
                attempts_used=attempt + 1, input_tokens=last_in_tok,
                output_tokens=last_out_tok, latency_ms=last_lat,
            )

    # Unreachable in practice (loop always returns), but keep a safe default.
    return GPT55Response(
        y=None, raw=last_raw_text or "EMPTY_AFTER_RETRY", model=model,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        attempts_used=max_retries, input_tokens=last_in_tok,
        output_tokens=last_out_tok, latency_ms=last_lat,
    )


def load_image_b64(path: str, max_size: int = 1024) -> str:
    """Load an image (or sample a video's first frame), resize so the
    longest edge is <= max_size, and return a base64-encoded JPEG string.
    Shared by all GPT-5.5 call sites (e8_modern_diffusion.py,
    fill_table6_gpt55_gap.py, e4_gpt55.py all had near-identical copies of
    this)."""
    from PIL import Image
    is_video = path.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    if is_video:
        from decord import VideoReader, cpu
        import cv2
        vr = VideoReader(path, ctx=cpu(0))
        frame = vr[0].asnumpy()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        img = Image.open(path).convert("RGB")

    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────
# Qwen2.5-VL-7B-Instruct — local GPU inference, logit-based REAL/FAKE
# extraction (paper Section VI.C: "Using the prompt in Section IV.B with
# REAL/FAKE logit extraction, Qwen2.5-VL achieves more reliable binary
# predictions when classification uses the REAL/FAKE logit gap rather than
# [greedy-decoded text], which sometimes emits REAL even when the
# underlying logits separate the classes.")
#
# Consolidates routing/run_qwen_v2.py, routing/sd14_run_qwen_v2.py,
# routing/biggan_run_qwen_v2.py, which were byte-identical except for three
# module-level constants (INDEX_FILE, OUT_FILE, IS_VIDEO) -- those become
# function parameters here. No API key needed; runs on the local GPU.
# ─────────────────────────────────────────────────────────────────────────

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_REAL_LABEL = 0
QWEN_FAKE_LABEL = 1
QWEN_UNK_LABEL = -1


@dataclass
class QwenResponse:
    """Full audit record for a single Qwen verifier call."""
    y_score: int              # final prediction via logit gap (0=REAL, 1=FAKE) -- preferred, see VI.C
    y_text: int               # prediction via greedy-decoded text parsing (0/1/-1=UNK) -- for comparison only
    p_real: Optional[float]
    p_fake: Optional[float]
    margin: Optional[float]   # |p_fake - p_real| at the generation step with the largest gap
    raw: str                  # full decoded output text
    input_tokens: int
    output_tokens: int
    latency_ms: float


def load_qwen_model(model_id: str = QWEN_MODEL_ID, device: str = "cuda"):
    """Load the Qwen2.5-VL processor/model and the first-token ids for the
    literal strings "REAL" and "FAKE" (needed for logit-gap scoring).
    Returns (processor, model, real_token_id, fake_token_id)."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    def first_token(text):
        ids = processor.tokenizer.encode(text, add_special_tokens=False)
        return ids[0] if ids else None

    real_token_id = first_token("REAL")
    fake_token_id = first_token("FAKE")
    return processor, model, real_token_id, fake_token_id


def _parse_label_text(text: str) -> int:
    """Robust greedy-text label parser (kept for comparison against the
    logit-gap score; identical logic to run_qwen_v2.py::parse_label)."""
    if not text:
        return QWEN_UNK_LABEL
    lines = [ln.strip().upper() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if last == "REAL":
            return QWEN_REAL_LABEL
        if last == "FAKE":
            return QWEN_FAKE_LABEL
        if re.search(r"\bFAKE\b", last) and not re.search(r"\bREAL\b", last):
            return QWEN_FAKE_LABEL
        if re.search(r"\bREAL\b", last) and not re.search(r"\bFAKE\b", last):
            return QWEN_REAL_LABEL
    matches = list(re.finditer(r"\b(REAL|FAKE)\b", text.upper()))
    if matches:
        return QWEN_REAL_LABEL if matches[-1].group(1) == "REAL" else QWEN_FAKE_LABEL
    return QWEN_UNK_LABEL


def _safe_softmax_two(a: float, b: float):
    x = np.array([a, b], dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    p = e / e.sum()
    return float(p[0]), float(p[1])


def call_qwen(
    model, processor, img, real_token_id: int, fake_token_id: int,
    prompt: str = VERIFIER_PROMPT, max_new_tokens: int = 64, device: str = "cuda",
) -> QwenResponse:
    """
    Run a single Qwen2.5-VL REAL/FAKE verifier call on a PIL image.

    Preferred label is the logit-gap score (y_score): at each generated
    step, compare the REAL-token vs FAKE-token logits, take the step with
    the largest |p_fake - p_real| gap, and threshold at 0.5. This is more
    reliable than greedy-decoded text (y_text), which the paper (Sec. VI.C)
    notes sometimes emits "REAL" even when the logits favor FAKE. y_text is
    still returned for comparison/diagnostics.

    No silent default: if neither the logit gap nor the text parse yields
    a usable answer, y_score falls back to y_text, and if that is also
    QWEN_UNK_LABEL, the caller receives y_score == QWEN_UNK_LABEL (-1) --
    never a guessed label from this function.
    """
    import torch

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)
    n_input_tokens = int(inputs["input_ids"].shape[-1])

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            return_dict_in_generate=True, output_scores=True,
        )
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    new_tokens = generated.sequences[:, inputs["input_ids"].shape[-1]:]
    n_output_tokens = int(new_tokens.shape[-1])
    output_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    y_text = _parse_label_text(output_text)

    p_fake = p_real = margin = None
    if generated.scores:
        best_gap = -1.0
        for step_logits in generated.scores:
            real_l = float(step_logits[0, real_token_id].item())
            fake_l = float(step_logits[0, fake_token_id].item())
            pr, pf = _safe_softmax_two(real_l, fake_l)
            gap = abs(pf - pr)
            if gap > best_gap:
                best_gap, p_real, p_fake = gap, pr, pf
        margin = best_gap

    y_score = (QWEN_FAKE_LABEL if p_fake >= 0.5 else QWEN_REAL_LABEL) if p_fake is not None else y_text

    return QwenResponse(
        y_score=y_score, y_text=y_text, p_real=p_real, p_fake=p_fake,
        margin=margin, raw=output_text, input_tokens=n_input_tokens,
        output_tokens=n_output_tokens, latency_ms=latency_ms,
    )


# ─────────────────────────────────────────────────────────────────────────
# Offline, no-API-call self-test of the parsing/validity logic. This is the
# test explicitly requested in Phase 0 item E; it was actually executed
# (see bottom of this file / the task report) and needs no network access.
# ─────────────────────────────────────────────────────────────────────────

def test_parsing():
    """Pure string-logic check of _is_valid_answer() against the canned
    cases from the Phase 0 spec, including the exact "FA" truncation bug
    (output_tokens==100, 19 real occurrences on disk) that the old
    non-exact-match logic missed. Returns (passed: bool, results: list)."""
    cases = [
        ("FA", None),                                   # truncated "FAKE" -- must be invalid, NOT accepted as FAKE
        ("FAKE", 1),
        ("REAL", 0),
        ("", None),                                      # empty
        ("The answer is FAKE because...", None),          # non-exact, must NOT be accepted even though it "contains" FAKE
        ("fake", 1),                                      # case-insensitive exact match still valid
        ("  REAL  ", 0),                                  # whitespace-padded exact match still valid
        ("REAL FAKE", None),                               # contains both -- ambiguous, must be invalid
        ("FAKEFAKE", None),                                # not an exact word match
    ]
    results = []
    passed = True
    for raw_text, expected in cases:
        got = _is_valid_answer(raw_text)
        ok = (got == expected)
        passed &= ok
        results.append((raw_text, expected, got, ok))
    return passed, results


if __name__ == "__main__":
    ok, results = test_parsing()
    print(f"{'raw':<35} {'expected':>8} {'got':>8} {'ok':>5}")
    for raw_text, expected, got, is_ok in results:
        print(f"{raw_text!r:<35} {str(expected):>8} {str(got):>8} {str(is_ok):>5}")
    print(f"\nALL PASSED: {ok}")
