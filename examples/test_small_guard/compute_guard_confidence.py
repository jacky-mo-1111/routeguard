#!/usr/bin/env python3
"""
Post-process guardrail generated_predictions.jsonl: add `confidence` per line.

At the last generated token, use softmax probabilities at token ids for " safe" vs " unsafe"
(default Qwen3Guard-Gen-0.6B: 6092 and 19860).  Then:

  confidence = P(correct_verdict_token) / (P(safe_token) + P(unsafe_token))

where `correct` is taken from the `label` field (ground truth safe/unsafe).

Usage:
  python compute_guard_confidence.py --from-yaml examples/test_small_guard/eval_agent.yaml
  python compute_guard_confidence.py --input-glob 'results/.../generated_predictions.jsonl' --model /path/to/ckpt
  python compute_guard_confidence.py --input-jsonl ... --model ... --batch-size 128
  python compute_guard_confidence.py ... --resume   # append after partial output (same input file)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Optional

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

VERDICT_RE = re.compile(r"Safety Assessment:\s*(safe|unsafe)", re.IGNORECASE | re.DOTALL)


def count_newlines_fast(path: str) -> int:
    """Buffered newline count for tqdm total (matches line iteration on text files)."""
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            n += chunk.count(b"\n")
    return n


def tqdm_lines(
    iterable,
    *,
    desc: str,
    leave: bool,
    total: Optional[int],
    initial: int = 0,
) -> tqdm:
    """Line iterator with a visible percentage bar when total is known."""
    disable = os.environ.get("TQDM_DISABLE", "").strip() == "1"
    kwargs: dict[str, Any] = {
        "desc": desc,
        "unit": "line",
        "leave": leave,
        "file": sys.stderr,
        "dynamic_ncols": True,
        "miniters": 1,
        "mininterval": 0.1,
        "disable": disable,
        "initial": initial,
    }
    if total is not None and total > 0:
        kwargs["total"] = total
    return tqdm(iterable, **kwargs)


def resolve_verdict_token_ids(tokenizer) -> tuple[int, int]:
    """Token ids for ' safe' and ' unsafe' after ':' in Safety Assessment (Qwen-style)."""
    if os.environ.get("GUARDRAIL_TOKEN_SAFE") and os.environ.get("GUARDRAIL_TOKEN_UNSAFE"):
        return int(os.environ["GUARDRAIL_TOKEN_SAFE"]), int(os.environ["GUARDRAIL_TOKEN_UNSAFE"])
    s = tokenizer.encode(" safe", add_special_tokens=False)
    u = tokenizer.encode(" unsafe", add_special_tokens=False)
    if not s or not u:
        raise ValueError("tokenizer could not encode ' safe' / ' unsafe'")
    return s[-1], u[-1]


def load_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def verdict_from_text(text: str) -> Optional[str]:
    m = VERDICT_RE.search(text)
    if not m:
        return None
    return m.group(1).lower()


def _ensure_pad_token(tokenizer) -> None:
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def prefix_ids_from_prompt_predict(tokenizer, prompt: str, predict: str) -> Optional[list[int]]:
    """Token ids for all but the last token of (prompt+predict); None if confidence cannot be computed."""
    if not predict or not predict.strip():
        return None
    full = prompt + predict
    ids = tokenizer.encode(full, add_special_tokens=False)
    if len(ids) < 2:
        return None
    return ids[:-1]


@torch.inference_mode()
def forward_last_logits_batch(
    model,
    tokenizer,
    prefixes: list[list[int]],
    device: torch.device,
) -> torch.Tensor:
    """Logits at the last prefix position per row, shape [B, vocab]. Right-padded batch."""
    if not prefixes:
        return torch.empty(0, 0, device=device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    lengths = [len(p) for p in prefixes]
    max_len = max(lengths)
    bsz = len(prefixes)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
    for i, p in enumerate(prefixes):
        L = len(p)
        input_ids[i, :L] = torch.tensor(p, dtype=torch.long, device=device)
        attn[i, :L] = 1
    out = model(input_ids, attention_mask=attn)
    logits = out.logits
    last_idx = torch.tensor([lengths[i] - 1 for i in range(bsz)], device=device, dtype=torch.long)
    batch_idx = torch.arange(bsz, device=device, dtype=torch.long)
    return logits[batch_idx, last_idx]


def confidence_for_gt(gt: str, denom: float, p_s: float, p_u: float) -> Optional[float]:
    if gt == "safe":
        return p_s / denom
    if gt == "unsafe":
        return p_u / denom
    return None


def process_jsonl(
    path_in: str,
    path_out: str,
    model,
    tokenizer,
    device: torch.device,
    token_safe: int,
    token_unsafe: int,
    *,
    batch_size: int = 32,
    line_tqdm_leave: bool = True,
    resume: bool = False,
) -> dict[str, int]:
    """Stream one output line per non-empty input line; flush so中断可 --resume 续跑。"""
    n = 0
    ok = 0
    missing_label_verdict = 0
    missing_conf = 0
    # (obj, gt, prefix_ids) — prefix_ids = token ids for all but last token of (prompt+predict)
    forward_buf: list[tuple[dict[str, Any], str, list[int]]] = []

    os.makedirs(os.path.dirname(path_out) or ".", exist_ok=True)
    out_exists = os.path.isfile(path_out)
    n_done = count_newlines_fast(path_out) if resume and out_exists else 0
    if resume and out_exists and n_done > 0:
        print(
            f"Resume: skipping first {n_done} non-empty records (existing {path_out})",
            file=sys.stderr,
        )
    elif resume and out_exists and n_done == 0:
        print(f"Resume: {path_out} empty, starting fresh", file=sys.stderr)

    out_mode = "a" if (resume and out_exists and n_done > 0) else "w"

    def emit_line(obj: dict[str, Any], out_fp) -> None:
        out_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
        out_fp.flush()

    def flush_forward_buf(out_fp) -> None:
        nonlocal forward_buf, ok, missing_conf
        while forward_buf:
            chunk = forward_buf[:batch_size]
            del forward_buf[:batch_size]
            prefixes = [p for _, _, p in chunk]
            last_logits = forward_last_logits_batch(model, tokenizer, prefixes, device)
            probs = F.softmax(last_logits, dim=-1)
            for j, (obj, gt, _) in enumerate(chunk):
                p_s = probs[j, token_safe].item()
                p_u = probs[j, token_unsafe].item()
                denom = p_s + p_u
                if denom <= 1e-12:
                    missing_conf += 1
                    obj["confidence"] = None
                    emit_line(obj, out_fp)
                    continue
                conf = confidence_for_gt(gt, denom, p_s, p_u)
                obj["confidence"] = round(conf, 6) if conf is not None else None
                ok += 1
                emit_line(obj, out_fp)

    line_desc = os.path.basename(path_in)
    line_total = count_newlines_fast(path_in)
    nonempty_idx = 0
    if not resume and out_exists:
        print(
            f"Warning: overwriting existing {path_out} (use --resume to append)",
            file=sys.stderr,
        )
    with open(path_out, out_mode, encoding="utf-8") as out_fp, open(path_in, encoding="utf-8") as in_fp:
        for line in tqdm_lines(
            in_fp,
            desc=line_desc,
            leave=line_tqdm_leave,
            total=line_total if line_total > 0 else None,
        ):
            line = line.strip()
            if not line:
                continue
            nonempty_idx += 1
            if nonempty_idx <= n_done:
                continue

            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            predict = obj.get("predict", "")
            label = obj.get("label", "")
            gt = verdict_from_text(label)
            if gt is None:
                flush_forward_buf(out_fp)
                missing_label_verdict += 1
                obj["confidence"] = None
                emit_line(obj, out_fp)
                n += 1
                continue

            pre = prefix_ids_from_prompt_predict(tokenizer, prompt, predict)
            if pre is None:
                flush_forward_buf(out_fp)
                missing_conf += 1
                obj["confidence"] = None
                emit_line(obj, out_fp)
                n += 1
                continue

            forward_buf.append((obj, gt, pre))
            if len(forward_buf) >= batch_size:
                flush_forward_buf(out_fp)
            n += 1

        flush_forward_buf(out_fp)

    return {
        "lines": n,
        "with_confidence": ok,
        "missing_label_verdict": missing_label_verdict,
        "missing_conf": missing_conf,
        "resumed_skipped": n_done,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-yaml", default=None, help="Training/eval yaml with model_name_or_path and optional guardrail_confidence")
    ap.add_argument("--input-glob", default=None, help="Glob for generated_predictions.jsonl")
    ap.add_argument("--input-jsonl", default=None, help="Single jsonl file")
    ap.add_argument("--model", default=None, help="Override model path")
    ap.add_argument("--output-suffix", default="_conf", help="Output name: generated_predictions{suffix}.jsonl")
    ap.add_argument("--force", action="store_true", help="Run even if guardrail_confidence is false in yaml")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for model forward (variable-length sequences are padded per batch).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If output jsonl exists, skip the first N non-empty input lines where N = its line count, and append.",
    )
    args = ap.parse_args()

    cfg: dict[str, Any] = {}
    if args.from_yaml:
        cfg = load_yaml(args.from_yaml)
        if not (args.force or cfg.get("guardrail_confidence", False)):
            print(
                "guardrail_confidence is not true in yaml; use --force or set guardrail_confidence: true",
                file=sys.stderr,
            )
            sys.exit(0)

    model_path = args.model or cfg.get("model_name_or_path")
    if not model_path:
        print("Need --model or --from-yaml with model_name_or_path", file=sys.stderr)
        sys.exit(1)

    paths: list[str] = []
    if args.input_jsonl:
        paths = [args.input_jsonl]
    elif args.input_glob:
        import glob

        paths = sorted(glob.glob(args.input_glob))
    elif args.from_yaml:
        out_dir = cfg.get("output_dir")
        if not out_dir:
            print("--from-yaml requires output_dir in yaml or use --input-glob / --input-jsonl", file=sys.stderr)
            sys.exit(1)
        import glob

        pattern = os.path.join(out_dir, "**", "generated_predictions.jsonl")
        paths = sorted(glob.glob(pattern, recursive=True))
        suf = args.output_suffix
        paths = [p for p in paths if not p.endswith(f"generated_predictions{suf}.jsonl")]

    if not paths:
        print("No input files found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {model_path} ...")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _ensure_pad_token(tok)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    token_safe, token_unsafe = resolve_verdict_token_ids(tok)
    print(f"Verdict token ids: safe={token_safe}, unsafe={token_unsafe}")
    print(f"batch_size={args.batch_size}")

    total_stats = {"files": 0, "lines": 0, "with_confidence": 0}
    # 多文件时保留外层 files 条；单文件时只显示行级 tqdm，避免 files 一直 0/1
    use_file_bar = len(paths) > 1
    if use_file_bar:
        path_iter = tqdm(
            paths,
            desc="files",
            total=len(paths),
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
            disable=os.environ.get("TQDM_DISABLE", "").strip() == "1",
        )
    else:
        path_iter = paths
    inner_line_bar_leave = not use_file_bar
    for path_in in path_iter:
        base = os.path.basename(path_in)
        if base != "generated_predictions.jsonl":
            continue
        dirn = os.path.dirname(path_in)
        path_out = os.path.join(dirn, f"generated_predictions{args.output_suffix}.jsonl")
        st = process_jsonl(
            path_in,
            path_out,
            model,
            tok,
            device,
            token_safe,
            token_unsafe,
            batch_size=max(1, args.batch_size),
            line_tqdm_leave=inner_line_bar_leave,
            resume=args.resume,
        )
        total_stats["files"] += 1
        total_stats["lines"] += st["lines"]
        total_stats["with_confidence"] += st["with_confidence"]
        print(f"  wrote {path_out}  ({st})")

    print("Done:", total_stats)


if __name__ == "__main__":
    main()
