"""
Baseline for TruthfulQA MCQ data attribution.

Train pools: tofu, chatdoctor, bever, wmdp, tqa  (output fields)
Test:        model-generated predictions (predict fields)
Ground truth: all test samples should be attributed to "tqa"

Methods: bm25, sbert, rouge, supervised_1b

Usage:
  python run_baseline_tqa_mcq.py --method all --model all
  python run_baseline_tqa_mcq.py --method rouge --model all
  python run_baseline_tqa_mcq.py --method supervised_1b --model all
"""

import argparse
import json
import os
import re
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATASET_INFO_PATH = "/nas02/jacky/Debug_LM/data/dataset_info.json"

TEST_PATHS = {
    "llama": "/nas02/jacky/Debug_LM/results/llama_tqa_mcq/generated_predictions.jsonl",
    "qwen": "/nas02/jacky/Debug_LM/results/qwen_tqa_mcq/generated_predictions.jsonl",
}

TRAIN_KEYS = {
    "tofu": "tofu_train",
    "chatdoctor": "chatdoctor_train",
    "bever": "bever_train",
    "wmdp": "wmdp_train",
    "tqa": "tqa_train",
}
POOL_NAMES = sorted(TRAIN_KEYS.keys())


def clean_text(text: str) -> str:
    text = re.sub(r"<think>\s*</think>\s*", "", text)
    text = re.sub(r"<[^>]*>", "", text)
    return text.strip()


def load_train_pools() -> Dict[str, List[str]]:
    with open(DATASET_INFO_PATH) as f:
        info = json.load(f)
    pools = {}
    for name, key in TRAIN_KEYS.items():
        path = info[key]["file_name"]
        with open(path) as f:
            data = json.load(f)
        pools[name] = [clean_text(item["output"]) for item in data if item.get("output")]
        print(f"  [pool] {name}: {len(pools[name])} outputs")
    return pools


def load_test_queries(model: str) -> List[str]:
    path = TEST_PATHS[model]
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                queries.append(clean_text(obj.get("predict", "")))
    print(f"  [test] {model}: {len(queries)} predictions from {path}")
    return queries


def run_bm25(pools: Dict[str, List[str]], queries: List[str]) -> List[str]:
    from rank_bm25 import BM25Okapi

    print("  Building BM25 indices ...")
    indices = {}
    for name in POOL_NAMES:
        tokenized = [doc.lower().split() for doc in pools[name]]
        indices[name] = BM25Okapi(tokenized)

    preds = []
    for q in tqdm(queries, desc="  BM25", unit="q"):
        tokens = q.lower().split()
        best_name, best_score = None, -float("inf")
        for name in POOL_NAMES:
            scores = indices[name].get_scores(tokens)
            mx = float(scores.max()) if len(scores) > 0 else -float("inf")
            if mx > best_score:
                best_score = mx
                best_name = name
        preds.append(best_name)
    return preds


def run_sbert(
    pools: Dict[str, List[str]],
    queries: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> List[str]:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    pool_embs = {}
    for name in POOL_NAMES:
        print(f"  Encoding pool {name} ({len(pools[name])} docs) ...")
        pool_embs[name] = model.encode(
            pools[name], batch_size=batch_size, show_progress_bar=True,
            convert_to_tensor=True, normalize_embeddings=True,
        )

    print(f"  Encoding {len(queries)} queries ...")
    q_emb = model.encode(
        queries, batch_size=batch_size, show_progress_bar=True,
        convert_to_tensor=True, normalize_embeddings=True,
    )

    max_per_pool = {}
    for name in POOL_NAMES:
        sims = torch.mm(q_emb, pool_embs[name].T)
        max_per_pool[name] = sims.max(dim=1).values

    stacked = torch.stack([max_per_pool[n] for n in POOL_NAMES], dim=1)
    pred_indices = stacked.argmax(dim=1).cpu().numpy()
    preds = [POOL_NAMES[i] for i in pred_indices]
    return preds


DATASET_NAMES = ["tofu", "chatdoctor", "bever", "wmdp", "tqa"]
NAME2LABEL = {n: i for i, n in enumerate(DATASET_NAMES)}
LABEL2NAME = {i: n for i, n in enumerate(DATASET_NAMES)}

# ── ROUGE-L ──
_rouge_scorer_global = None
_train_pools_global = None


def _init_rouge_worker(train_pools_dict):
    global _rouge_scorer_global, _train_pools_global
    from rouge_score import rouge_scorer as rs
    _rouge_scorer_global = rs.RougeScorer(["rougeL"], use_stemmer=True)
    _train_pools_global = train_pools_dict


def _rouge_attribute_one(query: str) -> str:
    scorer = _rouge_scorer_global
    pools = _train_pools_global
    best_name, best_score = None, -1.0
    for name in POOL_NAMES:
        for ref in pools[name]:
            score = scorer.score(ref, query)["rougeL"].fmeasure
            if score > best_score:
                best_score = score
                best_name = name
    return best_name


def run_rouge(pools: Dict[str, List[str]], queries: List[str]) -> List[str]:
    n_workers = min(cpu_count(), 32)
    with Pool(n_workers, initializer=_init_rouge_worker, initargs=(pools,)) as pool:
        preds = list(tqdm(
            pool.imap(_rouge_attribute_one, queries,
                      chunksize=max(1, len(queries) // n_workers)),
            total=len(queries), desc="  ROUGE", unit="q",
        ))
    return preds


# ── Supervised LoRA Classifier (Llama-3.2-1B) ──
ADAPTER_DIR = "/nas02/jacky/Debug_LM/Baselines/saves/llm_classifier"


def run_supervised_1b(
    queries: List[str],
    llm_model: str = "meta-llama/Llama-3.2-1B",
    max_length: int = 512,
    batch_size: int = 32,
) -> List[str]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel

    print(f"  Loading adapter from {ADAPTER_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForSequenceClassification.from_pretrained(
        llm_model, num_labels=len(DATASET_NAMES), torch_dtype=torch.bfloat16,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model = model.merge_and_unload()
    model = model.cuda().eval()

    all_preds = []
    for i in tqdm(range(0, len(queries), batch_size), desc="  Supervised", unit="batch"):
        batch_texts = queries[i:i + batch_size]
        enc = tokenizer(
            batch_texts, truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())

    return [LABEL2NAME[p] for p in all_preds]


def evaluate(preds: List[str], gt_label: str = "tqa") -> dict:
    from collections import Counter
    n = len(preds)
    correct = sum(1 for p in preds if p == gt_label)
    dist = Counter(preds)

    print(f"  Accuracy (-> {gt_label}): {correct}/{n} = {correct/n:.4f}")
    print(f"  Distribution: { {k: f'{v}/{n} ({v/n:.4f})' for k, v in sorted(dist.items())} }")

    return {
        "accuracy": correct / n,
        "correct": correct,
        "total": n,
        "distribution": {k: v / n for k, v in sorted(dist.items())},
    }


def main():
    parser = argparse.ArgumentParser(description="TruthfulQA MCQ baseline")
    parser.add_argument("--method", default="all",
                        choices=["bm25", "sbert", "rouge", "supervised_1b", "all"])
    parser.add_argument("--model", default="all", choices=["llama", "qwen", "all"])
    parser.add_argument("--sbert_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--output_dir", default="/nas02/jacky/Debug_LM/Baselines")
    args = parser.parse_args()

    methods = ["bm25", "sbert", "rouge", "supervised_1b"] if args.method == "all" else [args.method]
    models = ["llama", "qwen"] if args.model == "all" else [args.model]

    print(f"\n{'=' * 70}")
    print("TruthfulQA MCQ — Data Attribution Baseline")
    print(f"{'=' * 70}")

    need_pools = any(m in methods for m in ["bm25", "sbert", "rouge"])
    pools = load_train_pools() if need_pools else None

    all_results = {}
    for model_name in models:
        print(f"\n{'#' * 60}")
        print(f"# Model: {model_name}")
        print(f"{'#' * 60}")

        queries = load_test_queries(model_name)
        all_results[model_name] = {}

        for method in methods:
            print(f"\n--- {method.upper()} ---")
            t0 = time.time()

            if method == "bm25":
                preds = run_bm25(pools, queries)
            elif method == "sbert":
                preds = run_sbert(pools, queries, args.sbert_model)
            elif method == "rouge":
                preds = run_rouge(pools, queries)
            elif method == "supervised_1b":
                preds = run_supervised_1b(queries)

            elapsed = time.time() - t0
            metrics = evaluate(preds)
            metrics["elapsed_s"] = round(elapsed, 1)
            all_results[model_name][method] = metrics
            print(f"  Time: {elapsed:.1f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, "baseline_tqa_mcq.json")
    if os.path.exists(result_file):
        with open(result_file) as f:
            existing = json.load(f)
        for mn, meths in all_results.items():
            existing.setdefault(mn, {}).update(meths)
        all_results = existing
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    txt_file = os.path.join(args.output_dir, "baseline_tqa_mcq.txt")
    with open(txt_file, "w") as f:
        f.write(f"{'=' * 70}\n")
        f.write("TruthfulQA MCQ — Data Attribution Baseline\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Train pools: {', '.join(POOL_NAMES)}\n")
        f.write(f"Ground truth: tqa\n\n")
        for mn in sorted(all_results):
            for meth in sorted(all_results[mn]):
                m = all_results[mn][meth]
                dist_str = "  ".join(f"{k}={v:.4f}" for k, v in m["distribution"].items())
                f.write(f"{mn:8s} {meth:8s}  acc={m['accuracy']:.4f}  "
                        f"({m['correct']}/{m['total']}, {m['elapsed_s']}s)\n")
                f.write(f"                   dist: {dist_str}\n")
        f.write(f"\n{'=' * 70}\n")

    print(f"\nResults saved to {result_file}")
    print(f"Summary saved to {txt_file}")


if __name__ == "__main__":
    main()
