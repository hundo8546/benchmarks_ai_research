"""
routing.pipeline.batch_gpt55 — OpenAI Batch API support for bulk GPT-5.5
verifier calls (Batch tier: half the price of Standard; see
claude_context_notes/openaitokentable.txt, "Batch pricing data" table:
gpt-5.5 (<272K context) = $2.50 / $15.00 per 1M input/output tokens, vs
Standard $5.00 / $30.00).

IMPORTANT: there is no OPENAI_API_KEY in this environment. Every function
below that performs real network I/O (submit_batch, poll_batch_status,
download_batch_results) raises immediately and loudly if the key is absent
-- see _require_api_key(). This module was built and validated (JSONL
construction, parsing logic, the "missing key" guards) WITHOUT ever calling
the live API. Once a key is available, going live is:

    from routing.pipeline.batch_gpt55 import submit_batch
    batch_id = submit_batch("routing/results/batch_requests/<name>.jsonl",
                             description="<name>")
    # ... wait up to 24h, then:
    from routing.pipeline.batch_gpt55 import poll_batch_status, download_batch_results, parse_batch_results
    poll_batch_status(batch_id)                      # check status == "completed"
    download_batch_results(batch_id, "out.jsonl")
    df = parse_batch_results("out.jsonl", custom_id_map_path="<name>.custom_id_map.csv")

─────────────────────────────────────────────────────────────────────────
STEP 1 FINDING -- Batch API request format (confirmed, with evidence)
─────────────────────────────────────────────────────────────────────────
Question: is the Responses API endpoint ("/v1/responses") a valid `url`
for a batch request line, or does batch mode require the older Chat
Completions endpoint ("/v1/chat/completions") for image inputs?

ANSWER: "/v1/responses" IS valid and is what this module uses. Evidence:

  1. Installed `openai` Python package (pip show -> version 3.31.0, the
     current PyPI release as of 2026-10-15, installed fresh into an
     isolated inspection dir for this check -- NOT imported from memory).
     `openai.resources.batches.Batches.create()`'s `endpoint` parameter is
     typed:
         endpoint: Literal["/v1/responses", "/v1/chat/completions",
                            "/v1/embeddings", "/v1/completions",
                            "/v1/moderations", "/v1/images/generations",
                            "/v1/images/edits", "/v1/videos"]
     "/v1/responses" is explicitly in that Literal -- the SDK will refuse
     any other string at the type-check level and the server enforces the
     same set.

  2. Public docs (https://developers.openai.com/api/docs/guides/batch,
     fetched live -- internet access confirmed working via a plain HEAD to
     pypi.org/api.openai.com before fetching) list the SAME valid `url`
     values and give the exact JSONL request-line shape:
         {"custom_id": "request-1", "method": "POST",
          "url": "/v1/chat/completions", "body": {...}}
     (their inline example happens to show chat.completions, but the
     endpoint list on the same page includes "/v1/responses", and the
     page's Java/Ruby SDK examples create a batch with
     endpoint="/v1/responses" directly.) `body`'s parameters "match those
     of the underlying endpoint" per the docs -- i.e. the batch line's
     `body` is passed through to whichever endpoint `url` names, unchanged.

  3. Residual uncertainty (flagged, not guessed away): the fetched docs
     page's own worked *image*-input example uses /v1/moderations, not
     /v1/responses -- it does not show a fully worked image+Responses-API
     batch example. However, (a) the Responses API's synchronous image
     input format (`{"type": "input_image", "image_url": "data:image/..."}`
     inside an `input` message, exactly what verifiers.call_gpt55() already
     sends) is the SAME body shape whether the request is dispatched
     synchronously via client.responses.create(**body) or asynchronously
     by the batch worker POSTing `body` to `/v1/responses` on your behalf
     -- the docs state batch bodies are untouched passthroughs of the
     target endpoint's normal request body, and (b) the SDK type stub
     draws no distinction between text-only and multimodal /v1/responses
     bodies for batch eligibility. This module's documented recommendation:
     before submitting the full ~17,900-call set, submit ONE 1-request
     smoke-test batch (e.g. a single row from set (a)) and confirm its
     output_text comes back non-empty, before submitting the remaining
     rows. That smoke test cannot be run here (no API key) and is left as
     the first live-key action.

  4. Output-file line shape, confirmed from the SDK's generated type stubs
     (types/batch.py: `output_file_id`; the docs' retrieval example reads
     each output line as {"id", "custom_id", "response": {"status_code",
     "request_id", "body"}, "error"}). For a /v1/responses batch, `body`
     is the raw (unwrapped-by-SDK) Responses API JSON object -- it has NO
     client-side-only convenience field like `.output_text` (that is a
     Python `@property` on openai.types.responses.Response, computed from
     `output` -- see types/responses/response.py:555-569 in the installed
     SDK -- it does not exist in the raw JSON). This module's
     `_extract_output_text_from_responses_body()` reimplements that exact
     property logic by hand against the raw dict: for each item in
     body["output"] where item["type"] == "message", for each content
     block in item["content"] where content["type"] == "output_text",
     concatenate content["text"]. Token usage is read from
     body["usage"]["input_tokens"] / body["usage"]["output_tokens"]
     (openai.types.responses.response_usage.ResponseUsage field names,
     confirmed from the same installed SDK).

─────────────────────────────────────────────────────────────────────────
Token budget for batch calls (no escalating retry)
─────────────────────────────────────────────────────────────────────────
verifiers.call_gpt55() escalates [2048, 4096] across up to 5 synchronous
retries specifically so that a truncated/invalid response can be retried
with MORE budget. A batch call gets exactly one shot -- there is no
per-row retry loop inside a single batch (resubmitting a whole new batch
for a handful of stragglers would be wasteful; see "Resubmission note"
below). So each batch request line uses a SINGLE generous budget:

    BATCH_MAX_OUTPUT_TOKENS = 4096   # == max(verifiers.GPT_MAX_OUTPUT_TOKENS)

Reasoning: 4096 is already the top of the synchronous escalating ladder
that verifiers.py's correct implementation uses as its last resort for a
hard case, so giving every batch row that budget from the start (rather
than the ladder's initial 2048) removes the #1 known failure mode this
pipeline has already hit in production -- the exact 19 "FA" truncations at
output_tokens==100 in the old fill_table6_gpt55_gap.py / e4_gpt55.py
implementations (see verifiers.py's module docstring). The model virtually
always answers in 1-3 output tokens for this exact prompt ("REAL" or
"FAKE" is 1 token each at the GPT-5.5 tokenizer), so the realized cost at
4096 budget is driven entirely by ACTUAL output tokens used (which the
batch result reports per-row), not by the budget ceiling -- the ceiling
only matters for the rare verbose/hedging response, which is exactly the
failure mode we want headroom for since there is no retry.

─────────────────────────────────────────────────────────────────────────
Resubmission note (Phase 0 task spec, item 2a)
─────────────────────────────────────────────────────────────────────────
Batch mode has no per-row escalating retry. Any row that comes back
invalid/truncated/empty after a batch completes (y_verifier is None in
parse_batch_results()'s output) should NOT trigger a brand-new batch
submission for just the stragglers -- batch overhead (file upload, queue
time, up to 24h turnaround) is not worth it for what should be a tiny
count. Instead, collect_stragglers() returns the list of image paths that
need a retry, and the caller should feed those into the EXISTING
Standard-tier synchronous verifiers.call_gpt55() (which already has the
correct escalating-budget + no-silent-default behavior) for a quick
synchronous mop-up pass. This is cheaper in wall-clock time and simpler
than a second batch for what should be at most a few dozen rows out of
~17,900.

─────────────────────────────────────────────────────────────────────────
No live API calls were made anywhere in building or testing this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import pandas as pd

from pipeline.verifiers import VERIFIER_PROMPT, _is_valid_answer, load_image_b64

# Same model id used throughout the rest of this pipeline's GPT-5.5 call
# sites (verifiers.call_gpt55's callers, fill_table6_gpt55_gap.py's MODEL
# constant, experiments/e8_modern_diffusion.py). Not exported as a
# module-level constant by verifiers.py itself (it's a parameter of
# call_gpt55()), so it is re-declared here, matching those call sites
# exactly rather than guessed.
GPT55_MODEL_ID = "gpt-5.5"

# Single generous one-shot budget for batch rows -- see module docstring
# "Token budget for batch calls" above for why 4096 (not the full
# escalating [2048, 4096] ladder) is correct here.
BATCH_MAX_OUTPUT_TOKENS = 4096

BATCH_ENDPOINT = "/v1/responses"  # see module docstring STEP 1 FINDING
BATCH_COMPLETION_WINDOW = "24h"   # only value the API currently supports


def _stable_custom_id(image_path: str) -> str:
    """Deterministic custom_id derived from an image path (sha1 hex, 40
    chars -- well under any reasonable custom_id length limit while still
    being effectively collision-free for a dataset of this size). Same
    path always maps to the same custom_id, so results can be joined back
    to the source row unambiguously after the batch completes."""
    return hashlib.sha1(image_path.encode("utf-8")).hexdigest()


def _require_api_key() -> str:
    """Raise immediately and clearly if OPENAI_API_KEY is unset/empty.
    Every function in this module that would perform real network I/O
    calls this FIRST, before doing anything else, so a missing key never
    results in a silent no-op."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set (or is empty) in the environment. "
            "Refusing to proceed -- this function performs real, billed "
            "OpenAI API calls and must not silently no-op. Set "
            "OPENAI_API_KEY and retry."
        )
    return api_key


def build_batch_request_file(rows: list[dict], out_path: str) -> dict:
    """
    Build a Batch-API JSONL request file for GPT-5.5 image verification.

    rows: list of {"custom_id": <unique str>, "image_path": <str>}.
          If a row's "custom_id" is missing or falsy, one is derived
          deterministically from "image_path" via _stable_custom_id() --
          callers of the step-3 request-set builder below always pass an
          explicit custom_id derived the same way, but this fallback keeps
          the function safe to call directly with bare (path-only) rows.

    Writes one JSONL line per successfully-loaded row to out_path, in the
    confirmed batch format (see module docstring):
        {"custom_id": ..., "method": "POST", "url": "/v1/responses",
         "body": {"model": "gpt-5.5", "input": [...], "max_output_tokens": 4096}}
    using the SAME prompt text (verifiers.VERIFIER_PROMPT) and the SAME
    image-loading/resizing/base64-JPEG logic (verifiers.load_image_b64,
    max edge 1024px, JPEG quality 90) as the synchronous call_gpt55().

    Also writes a companion custom_id -> image_path (+ y_true, if present
    on the row) mapping CSV next to out_path, at
    "<out_path without .jsonl>.custom_id_map.csv", since the batch
    OUTPUT file (once downloaded) will only carry custom_id back -- not
    the original path/label -- and parse_batch_results() needs this map
    to reconstruct the path/y_true columns.

    Rows whose image file is missing or fails to load are SKIPPED (not
    written to the JSONL, not written to the map), with a warning printed
    for each, and are counted in the returned summary.

    Returns a summary dict: {
        "out_path", "custom_id_map_path", "n_rows_requested", "n_rows_written",
        "n_rows_skipped", "skipped": [{"custom_id","image_path","reason"}, ...],
        "total_base64_chars", "jsonl_file_size_bytes",
    }
    """
    skipped = []
    n_written = 0
    total_b64_chars = 0

    map_path = _custom_id_map_path(out_path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w") as jf, open(map_path, "w") as mf:
        mf.write("custom_id,image_path,y_true\n")
        for row in rows:
            image_path = row.get("image_path") or row.get("path")
            custom_id = row.get("custom_id") or (_stable_custom_id(image_path) if image_path else None)
            y_true = row.get("y_true", "")

            if not image_path:
                reason = "missing image_path in row"
                print(f"  WARNING: skipping row (custom_id={custom_id!r}): {reason}")
                skipped.append({"custom_id": custom_id, "image_path": image_path, "reason": reason})
                continue
            if not os.path.exists(image_path):
                reason = "file not found"
                print(f"  WARNING: skipping {image_path!r} (custom_id={custom_id}): {reason}")
                skipped.append({"custom_id": custom_id, "image_path": image_path, "reason": reason})
                continue

            try:
                b64 = load_image_b64(image_path)
            except Exception as e:  # noqa: BLE001 -- any decode/load failure should be a skip, not a crash
                reason = f"unreadable/corrupt image: {e!r}"
                print(f"  WARNING: skipping {image_path!r} (custom_id={custom_id}): {reason}")
                skipped.append({"custom_id": custom_id, "image_path": image_path, "reason": reason})
                continue

            total_b64_chars += len(b64)

            body = {
                "model": GPT55_MODEL_ID,
                "input": [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": VERIFIER_PROMPT},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                    ],
                }],
                "max_output_tokens": BATCH_MAX_OUTPUT_TOKENS,
            }
            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": BATCH_ENDPOINT,
                "body": body,
            }
            jf.write(json.dumps(line) + "\n")
            mf.write(f"{custom_id},{image_path},{y_true}\n")
            n_written += 1

    jsonl_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0

    summary = {
        "out_path": out_path,
        "custom_id_map_path": map_path,
        "n_rows_requested": len(rows),
        "n_rows_written": n_written,
        "n_rows_skipped": len(skipped),
        "skipped": skipped,
        "total_base64_chars": total_b64_chars,
        "jsonl_file_size_bytes": jsonl_size,
    }
    print(f"build_batch_request_file: wrote {n_written}/{len(rows)} rows to {out_path} "
          f"({jsonl_size:,} bytes); skipped {len(skipped)}; map -> {map_path}")
    return summary


def _custom_id_map_path(out_path: str) -> str:
    if out_path.endswith(".jsonl"):
        return out_path[: -len(".jsonl")] + ".custom_id_map.csv"
    return out_path + ".custom_id_map.csv"


BATCH_MAX_FILE_BYTES = 200 * 1000 * 1000  # OpenAI Batch API limit: input file <= 200 MB
                                           # (confirmed via developers.openai.com/api/docs/guides/batch,
                                           # fetched live -- see module docstring STEP 1 FINDING).
                                           # 50,000 requests/batch is the OTHER limit; every request
                                           # set built here is far under that count, so file SIZE (not
                                           # row count) is always the binding constraint for this
                                           # pipeline's large image payloads.


def split_batch_request_file(jsonl_path: str, max_bytes: int = BATCH_MAX_FILE_BYTES - 5_000_000) -> list[str]:
    """
    Split an oversized batch-request JSONL (built by build_batch_request_file)
    into multiple part files, each safely under the Batch API's 200MB
    per-file limit (max_bytes defaults to 195MB, a 5MB safety margin below
    the hard 200MB cap). Base64-encoded JPEG images make these request
    files large -- several of this project's request sets (original_voc_union,
    coco_union, full_sdxl_flux) exceed 200MB as a single file and MUST be
    split before submit_batch() can be called on them; only the smallest
    set (bugfix_reruns, ~90MB) fits in one file already.

    If jsonl_path is already <= max_bytes, returns [jsonl_path] unchanged
    (no part files are created) -- callers can always treat the return
    value as "the list of files to pass to submit_batch(), one call per
    file," regardless of whether splitting happened.

    Splitting is purely at the JSONL-file level (greedy: accumulate whole
    lines until the next line would exceed max_bytes, then start a new
    part) -- custom_id values are untouched, and the SAME single
    custom_id_map.csv companion file (written once by
    build_batch_request_file) still covers every part, since joining
    results back to path/y_true is done by custom_id, not by which file a
    request happened to land in.

    Part files are named "<stem>_part1.jsonl", "<stem>_part2.jsonl", ...
    next to the original file. The original (oversized) file is left
    in place, unmodified -- the parts are additional files, not a
    replacement, so nothing downstream that already referenced the
    original path breaks.
    """
    size = os.path.getsize(jsonl_path)
    if size <= max_bytes:
        return [jsonl_path]

    stem = jsonl_path[:-len(".jsonl")] if jsonl_path.endswith(".jsonl") else jsonl_path
    part_paths: list[str] = []
    part_num = 1
    cur_bytes = 0
    cur_lines: list[str] = []

    def _flush():
        nonlocal part_num, cur_bytes, cur_lines
        if not cur_lines:
            return
        part_path = f"{stem}_part{part_num}.jsonl"
        with open(part_path, "w") as pf:
            pf.writelines(cur_lines)
        part_paths.append(part_path)
        part_num += 1
        cur_bytes = 0
        cur_lines = []

    with open(jsonl_path) as f:
        for line in f:
            line_bytes = len(line.encode("utf-8"))
            if cur_lines and cur_bytes + line_bytes > max_bytes:
                _flush()
            cur_lines.append(line)
            cur_bytes += line_bytes
    _flush()

    total_part_bytes = sum(os.path.getsize(p) for p in part_paths)
    print(f"split_batch_request_file: {jsonl_path} ({size:,} bytes) > {max_bytes:,} byte limit -- "
          f"split into {len(part_paths)} part(s) ({total_part_bytes:,} bytes total): {part_paths}")
    return part_paths


def submit_batch(request_file_path: str, description: str) -> str:
    """
    Upload a batch request JSONL file and submit it as a Batch-API job.

    MUST have a real, non-empty OPENAI_API_KEY in the environment --
    raises RuntimeError immediately otherwise (see _require_api_key()).
    This function cannot be exercised end-to-end in this environment (no
    key); only the "raises when key is absent" behavior has been run and
    verified here.

    Usage (once a key is available):
        batch_id = submit_batch(
            "routing/results/batch_requests/bugfix_reruns.jsonl",
            description="bugfix reruns: 19 FA truncations + fill_table6 backfill",
        )
        # batch_id looks like "batch_abc123..."; save it -- you need it for
        # poll_batch_status() / download_batch_results().

    IMPORTANT -- 200MB per-file limit: the OpenAI Batch API caps each
    uploaded input file at 200MB. This project's base64-JPEG image
    payloads make several request sets exceed that as a single file (see
    split_batch_request_file()). For any file over the limit, call
    split_batch_request_file() FIRST and call submit_batch() once per part
    (each part's rows all carry the SAME metadata["description"] plus a
    "_partN" suffix recommended for clarity):
        for i, part in enumerate(split_batch_request_file(path), start=1):
            submit_batch(part, description=f"{description} (part {i})")
    This function itself does not auto-split -- it submits exactly the one
    file given to it, and raises (via the server's own file-size
    validation) if that file is still over 200MB.

    Returns the new batch's id (str).
    """
    api_key = _require_api_key()
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    with open(request_file_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint=BATCH_ENDPOINT,
        completion_window=BATCH_COMPLETION_WINDOW,
        metadata={"description": description[:512]},
    )
    print(f"submit_batch: submitted {request_file_path} -> batch_id={batch.id} "
          f"(input_file_id={batch_input_file.id}, status={batch.status})")
    return batch.id


def poll_batch_status(batch_id: str) -> dict:
    """
    Retrieve the current status of a batch. MUST have a real
    OPENAI_API_KEY -- raises immediately otherwise.

    Returns a dict with the fields most relevant for deciding what to do
    next: status, request_counts (completed/failed/total), output_file_id,
    error_file_id, created_at, completed_at, failed_at, expires_at.
    """
    api_key = _require_api_key()
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    batch = client.batches.retrieve(batch_id)
    counts = batch.request_counts
    return {
        "id": batch.id,
        "status": batch.status,
        "endpoint": batch.endpoint,
        "request_counts_completed": getattr(counts, "completed", None),
        "request_counts_failed": getattr(counts, "failed", None),
        "request_counts_total": getattr(counts, "total", None),
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "created_at": batch.created_at,
        "completed_at": batch.completed_at,
        "failed_at": batch.failed_at,
        "expired_at": batch.expired_at,
        "expires_at": batch.expires_at,
    }


def download_batch_results(batch_id: str, out_path: str) -> str:
    """
    Download a completed batch's output JSONL to out_path. MUST have a
    real OPENAI_API_KEY -- raises immediately otherwise. Also raises if
    the batch has no output_file_id yet (not completed, or completed with
    zero successes -- check poll_batch_status()'s error_file_id in that
    case, which carries per-row errors and is NOT downloaded by this
    function).

    Returns out_path.
    """
    api_key = _require_api_key()
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    batch = client.batches.retrieve(batch_id)
    if not batch.output_file_id:
        raise RuntimeError(
            f"Batch {batch_id} has no output_file_id yet (status={batch.status}). "
            f"Poll with poll_batch_status() until status == 'completed' before downloading. "
            f"If status is 'completed' but output_file_id is still empty, check "
            f"error_file_id={batch.error_file_id!r} -- every row may have failed."
        )
    content = client.files.content(batch.output_file_id)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(content.read() if hasattr(content, "read") else content.content)
    print(f"download_batch_results: saved batch {batch_id}'s output to {out_path}")
    return out_path


def _extract_output_text_from_responses_body(body: dict) -> str:
    """
    Reimplementation, against a raw JSON dict, of
    openai.types.responses.response.Response.output_text's exact logic
    (installed SDK, types/responses/response.py:555-569): concatenate the
    "text" field of every content block of type "output_text", inside
    every output item of type "message". This convenience property exists
    client-side only on the SDK's parsed Response object -- a raw batch
    output line's body is the unparsed JSON and has no such field, so it
    must be recomputed here.
    """
    texts = []
    for item in body.get("output") or []:
        if item.get("type") == "message":
            for content in item.get("content") or []:
                if content.get("type") == "output_text":
                    text = content.get("text")
                    if text is not None:
                        texts.append(text)
    return "".join(texts)


def parse_batch_results(results_jsonl_path: str, custom_id_map_path: Optional[str] = None) -> pd.DataFrame:
    """
    Parse a completed batch's downloaded output JSONL into the same
    column schema as the existing *_gpt55_*_disagree_preds.csv files:
    path, y_true, y_verifier, raw, latency_ms, input_tokens, output_tokens
    (confirmed via `pd.read_csv("gpt55_disagree_preds.csv", nrows=0).columns`
    -- note some sibling files, e.g. sdxl_gpt55_siglip2_disagree_preds.csv,
    use "y_gpt55" instead of "y_verifier" for the same meaning; this
    function standardizes on "y_verifier" per the task spec).

    An 8th column, "custom_id", is appended after those 7 for audit/join
    traceability -- the base 7 columns match the existing schema exactly
    in name and order; custom_id is an intentional addition, not part of
    the original schema, documented here rather than silently added.

    latency_ms is ALWAYS NaN for every row: batch mode has no per-call
    wall-clock concept (requests are processed asynchronously, in
    arbitrary order, over up to 24h) -- fabricating a number here would
    misrepresent real latency data from the synchronous pipeline, so this
    column is left NaN with this note instead.

    y_verifier uses verifiers.py's strict `_is_valid_answer()` -- None
    (never a guessed 0/1) for anything that is not EXACTLY "REAL" or
    "FAKE" after .strip().upper(), exactly like the synchronous path. A
    row with a batch-level `error` (request failed entirely) is also
    y_verifier=None, with raw set to "ERROR:<error details>".

    custom_id_map_path: optional path to a CSV with columns
    (custom_id, image_path, y_true) -- the companion file
    build_batch_request_file() writes next to its output JSONL. If given,
    "path" and "y_true" are joined in from this map. If omitted, "path"
    and "y_true" are left as None/NaN (with a printed warning) -- the
    caller must then join them in separately using the custom_id column,
    e.g. from the MANIFEST / custom_id_map.csv saved when the request set
    was built.
    """
    id_map = None
    if custom_id_map_path is not None:
        id_map = pd.read_csv(custom_id_map_path, dtype={"custom_id": str})
        id_map = id_map.set_index("custom_id")

    rows = []
    with open(results_jsonl_path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id")
            error = obj.get("error")
            response = obj.get("response") or {}
            body = response.get("body") or {}

            if error is not None:
                raw = f"ERROR:{error}"
                y_verifier = None
                in_tok = out_tok = 0
            elif not body:
                raw = "ERROR:NO_RESPONSE_BODY"
                y_verifier = None
                in_tok = out_tok = 0
            else:
                raw = _extract_output_text_from_responses_body(body)
                y_verifier = _is_valid_answer(raw)
                usage = body.get("usage") or {}
                in_tok = usage.get("input_tokens", 0) or 0
                out_tok = usage.get("output_tokens", 0) or 0

            path, y_true = None, None
            if id_map is not None:
                if custom_id in id_map.index:
                    path = id_map.loc[custom_id, "image_path"]
                    y_true = id_map.loc[custom_id, "y_true"]
                else:
                    print(f"  WARNING (line {line_no}): custom_id {custom_id!r} not found in "
                          f"custom_id_map_path {custom_id_map_path!r} -- path/y_true left as None")

            rows.append({
                "path": path,
                "y_true": y_true,
                "y_verifier": y_verifier,
                "raw": raw,
                "latency_ms": float("nan"),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "custom_id": custom_id,
            })

    if id_map is None and rows:
        print("  NOTE: no custom_id_map_path given -- 'path'/'y_true' columns are None for all "
              "rows; join them in separately using 'custom_id' before treating this like a "
              "*_gpt55_*_disagree_preds.csv file.")

    cols = ["path", "y_true", "y_verifier", "raw", "latency_ms", "input_tokens", "output_tokens", "custom_id"]
    return pd.DataFrame(rows, columns=cols)


def collect_stragglers(parsed_df: pd.DataFrame) -> list[str]:
    """
    Return the list of image paths needing a synchronous Standard-tier
    retry via verifiers.call_gpt55() (see module docstring's
    "Resubmission note") -- every row where y_verifier is null (invalid,
    truncated, empty, or an outright request error after the batch
    completed).

    Rows with a null path (no custom_id_map was supplied to
    parse_batch_results, so the path could not be recovered) cannot be
    turned into a retry list by path and are reported via a warning
    rather than silently dropped or silently included as None.
    """
    mask = parsed_df["y_verifier"].isna()
    stragglers = parsed_df.loc[mask]
    n_no_path = int(stragglers["path"].isna().sum())
    if n_no_path:
        print(f"  WARNING: {n_no_path} straggler row(s) have no recoverable path (parse_batch_results "
              f"was called without custom_id_map_path) -- they are excluded from the returned list. "
              f"Re-run parse_batch_results with custom_id_map_path set to recover them by path.")
    return stragglers["path"].dropna().tolist()
