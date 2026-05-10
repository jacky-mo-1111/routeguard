#!/usr/bin/env python3
"""Benchmark HF chat models on local JSON datasets: sequential generate timing."""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_dataset_rows(path: Path, n: int, seed: int, random_pick: bool) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array in {path}")
    take = min(n, len(rows))
    if random_pick:
        rnd = random.Random(seed)
        return rnd.sample(rows, take)
    return rows[:take]


def build_user_content(ex: dict) -> str:
    ins = ex.get("instruction") or ""
    inp = ex.get("input") or ""
    parts = [p for p in (ins.strip(), inp.strip()) if p]
    return "\n".join(parts) if parts else ""


def warmup(model, tokenizer, device: str, max_new_tokens: int) -> None:
    msgs = [{"role": "user", "content": "Say OK."}]
    try:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = {k: v.to(device) for k, v in tokenizer(text, return_tensors="pt").items()}
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=min(16, max_new_tokens), pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.inference_mode()
def run_benchmark(
    model_id: str,
    rows: list[dict],
    max_new_tokens: int,
    device: str,
    dtype_str: str,
) -> tuple[float, list[float]]:
    if device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    model.eval()

    use_chat_template = getattr(tokenizer, "chat_template", None) is not None
    warmup(model, tokenizer, device, max_new_tokens)

    latencies: list[float] = []
    wall0 = time.perf_counter()

    for ex in rows:
        user_txt = build_user_content(ex)
        if not user_txt:
            latencies.append(0.0)
            continue

        if use_chat_template:
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_txt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_txt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prompt = f"User:\n{user_txt}\n\nAssistant:\n"

        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        enc = {k: v.to(device) for k, v in enc.items()}
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    wall1 = time.perf_counter()
    total = wall1 - wall0

    del model
    del tokenizer
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return total, latencies


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_info", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", required=True, help="Keys in dataset_info.json")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--random_sample",
        action="store_true",
        help="randomly choose N rows (seeded); default is first N rows in file order.",
    )
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--out_json", type=Path, default=None)
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    info = json.loads(args.dataset_info.read_text(encoding="utf-8"))

    results: list[dict] = []
    for ds_key in args.datasets:
        if ds_key not in info:
            raise SystemExit(f"Unknown dataset_info key: {ds_key}")
        ds_path = Path(info[ds_key]["file_name"])
        if not ds_path.is_file():
            raise SystemExit(f"Dataset file missing: {ds_path}")
        rows = load_dataset_rows(ds_path, args.num_samples, args.seed, args.random_sample)
        n_effective = len(rows)

        for model_id in args.models:
            print(f"[run] model={model_id} dataset={ds_key} file={ds_path} n={n_effective}", flush=True)
            total_sec, latencies = run_benchmark(
                model_id, rows, args.max_new_tokens, args.device, args.dtype
            )
            summed = sum(latencies)
            mean_lat = total_sec / n_effective if n_effective else 0.0
            samples_s = n_effective / total_sec if total_sec > 0 else 0.0
            rec = {
                "model_name_or_path": model_id,
                "dataset_key": ds_key,
                "dataset_path": str(ds_path),
                "num_samples": n_effective,
                "random_sample": args.random_sample,
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "device": args.device,
                "dtype": args.dtype,
                "wall_total_sec": round(total_sec, 4),
                "sum_sample_latencies_sec": round(summed, 4),
                "mean_latency_sec": round(mean_lat, 6),
                "samples_per_sec": round(samples_s, 4),
            }
            results.append(rec)

    # Print table
    hdr = (
        f"{'model':<38} {'dataset':<20} {'n':>5} {'wall_s':>9} {'mean_lat_s':>12} {'samples/s':>10}"
    )
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        short_m = r["model_name_or_path"].split("/")[-1][:36]
        print(
            f"{short_m:<38} {r['dataset_key']:<20} {r['num_samples']:>5} "
            f"{r['wall_total_sec']:>9.2f} {r['mean_latency_sec']:>12.6f} {r['samples_per_sec']:>10.4f}"
        )

    payload = {"results": results}
    txt = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(txt + "\n", encoding="utf-8")
        print(f"\nWrote {args.out_json}")
    else:
        print("\n(raw json)")
        print(txt)


if __name__ == "__main__":
    main()
