"""
MCQ Data Attribution Baseline — per-dataset evaluation.

For each MCQ test subset (tofu_mcq, chatdoctor_mcq, beveratails_mcq, tqa_mcq),
attribute model predictions to one of 5 training pools using BM25/SBERT/ROUGE/supervised_1b.

Also counts DebugLM <TAG> accuracy from the predictions themselves.

Usage:
  python run_baseline_mcq.py --method all --model qwen
  python run_baseline_mcq.py --method bm25,sbert --model llama
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import Dict, List

import numpy as np
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATASET_INFO_PATH = "/nas02/jacky/Debug_LM/data/dataset_info.json"

TRAIN_KEYS = {
    "tofu": "tofu_train",
    "chatdoctor": "chatdoctor_train",
    "bever": "bever_train",
    "wmdp": "wmdp_train",
    "tqa": "tqa_train",
}
POOL_NAMES = sorted(TRAIN_KEYS.keys())

DATASET_NAMES = ["tofu", "chatdoctor", "bever", "wmdp", "tqa"]
NAME2LABEL = {n: i for i, n in enumerate(DATASET_NAMES)}
LABEL2NAME = {i: n for i, n in enumerate(DATASET_NAMES)}

MCQ_SUBSETS = {
    "tofu_mcq": {"gt": "tofu", "tag": "TOFU"},
    "chatdoctor_mcq": {"gt": "chatdoctor", "tag": "CHATDOCTOR"},
    "beveratails_mcq": {"gt": "bever", "tag": "BEVER"},
    "tqa_mcq": {"gt": "tqa", "tag": "TQA"},
}

MODEL_MCQ_DIRS = {
    "qwen": "/nas02/jacky/Debug_LM/results/qwen_mcq",
    "llama": "/nas02/jacky/Debug_LM/results/llama_mcq",
}

MODEL_TQA_MCQ_PATHS = {
    "llama": "/nas02/jacky/Debug_LM/results/llama_tqa_mcq/generated_predictions.jsonl",
    "qwen": "/nas02/jacky/Debug_LM/results/qwen_tqa_mcq/generated_predictions.jsonl",
}

MODEL_CD_MCQ_PATHS = {
    "llama": "/nas02/jacky/Debug_LM/results/llama_mcq_cd/generated_predictions.jsonl",
    "qwen": "/nas02/jacky/Debug_LM/results/qwen_mcq_cd/generated_predictions.jsonl",
}


def clean_text(text: str) -> str:
    text = re.sub(r"<think>\s*</think>\s*", "", text)
    text = re.sub(r"<[^>]*>", "", text)
    return text.strip()


def extract_question(prompt: str) -> str:
    """Extract the question text from a chat-template prompt, stripping all <> special tokens."""
    text = re.sub(r"<[^>]*>", "", prompt)
    text = re.sub(r"^\s*(user|assistant)\s*", "", text, flags=re.MULTILINE)
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


def load_mcq_predictions(model: str, subset: str) -> List[dict]:
    if subset == "tqa_mcq":
        path = MODEL_TQA_MCQ_PATHS.get(model)
        if path is None or not os.path.exists(path):
            return []
    elif subset == "chatdoctor_mcq":
        path = MODEL_CD_MCQ_PATHS.get(model)
        if path is None or not os.path.exists(path):
            base = MODEL_MCQ_DIRS.get(model)
            if base is None or not os.path.isdir(base):
                return []
            path = os.path.join(base, subset, "generated_predictions.jsonl")
    else:
        base = MODEL_MCQ_DIRS.get(model)
        if base is None or not os.path.isdir(base):
            return []
        path = os.path.join(base, subset, "generated_predictions.jsonl")
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def run_bm25(pools, queries):
    from rank_bm25 import BM25Okapi
    indices = {}
    for name in POOL_NAMES:
        tokenized = [doc.lower().split() for doc in pools[name]]
        indices[name] = BM25Okapi(tokenized)
    preds = []
    for q in tqdm(queries, desc="    BM25", unit="q"):
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


def run_sbert(pools, queries, sbert_model="all-MiniLM-L6-v2", batch_size=256):
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(sbert_model, device=device)
    pool_embs = {}
    for name in POOL_NAMES:
        pool_embs[name] = model.encode(
            pools[name], batch_size=batch_size, show_progress_bar=False,
            convert_to_tensor=True, normalize_embeddings=True,
        )
    q_emb = model.encode(
        queries, batch_size=batch_size, show_progress_bar=False,
        convert_to_tensor=True, normalize_embeddings=True,
    )
    max_per_pool = {}
    for name in POOL_NAMES:
        sims = torch.mm(q_emb, pool_embs[name].T)
        max_per_pool[name] = sims.max(dim=1).values
    stacked = torch.stack([max_per_pool[n] for n in POOL_NAMES], dim=1)
    pred_indices = stacked.argmax(dim=1).cpu().numpy()
    return [POOL_NAMES[i] for i in pred_indices]


_rouge_scorer_global = None
_train_pools_global = None


def _init_rouge_worker(pools):
    global _rouge_scorer_global, _train_pools_global
    from rouge_score import rouge_scorer as rs
    _rouge_scorer_global = rs.RougeScorer(["rougeL"], use_stemmer=True)
    _train_pools_global = pools


def _rouge_one(query):
    scorer = _rouge_scorer_global
    pools = _train_pools_global
    best_name, best_score = None, -1.0
    for name in POOL_NAMES:
        for ref in pools[name]:
            s = scorer.score(ref, query)["rougeL"].fmeasure
            if s > best_score:
                best_score = s
                best_name = name
    return best_name


def run_rouge(pools, queries):
    n_workers = min(cpu_count(), 32)
    with Pool(n_workers, initializer=_init_rouge_worker, initargs=(pools,)) as pool:
        preds = list(tqdm(
            pool.imap(_rouge_one, queries, chunksize=max(1, len(queries) // n_workers)),
            total=len(queries), desc="    ROUGE", unit="q",
        ))
    return preds


ADAPTER_DIR = "/nas02/jacky/Debug_LM/Baselines/saves/llm_classifier"


def run_supervised_1b(queries, llm_model="meta-llama/Llama-3.2-1B", max_length=512, batch_size=32):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model = AutoModelForSequenceClassification.from_pretrained(
        llm_model, num_labels=len(DATASET_NAMES), torch_dtype=torch.bfloat16,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model = model.merge_and_unload().cuda().eval()
    all_preds = []
    for i in tqdm(range(0, len(queries), batch_size), desc="    Supervised", unit="batch"):
        batch = queries[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return [LABEL2NAME[p] for p in all_preds]


def evaluate(preds, gt_label):
    n = len(preds)
    correct = sum(1 for p in preds if p == gt_label)
    dist = Counter(preds)
    return {
        "accuracy": correct / n if n > 0 else 0,
        "correct": correct,
        "total": n,
        "distribution": {k: v / n for k, v in sorted(dist.items())},
    }


def debuglm_accuracy(entries, gt_tag):
    total = len(entries)
    tag_counts = Counter()
    for e in entries:
        tags = re.findall(r"<TAG><(\w+)>", e.get("predict", ""))
        for t in tags:
            tag_counts[t] += 1
    correct = tag_counts.get(gt_tag, 0)
    return {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "distribution": dict(tag_counts),
    }


def mcq_accuracy(entries):
    total = len(entries)
    correct = 0
    for e in entries:
        pred_text = re.sub(r"<[^>]*>", "", e.get("predict", "")).strip()
        label_text = re.sub(r"<[^>]*>", "", e.get("label", "")).strip()
        pred_letter = pred_text[:1].upper() if pred_text else ""
        label_letter = label_text[:1].upper() if label_text else ""
        if pred_letter and pred_letter == label_letter:
            correct += 1
    return {"accuracy": correct / total if total > 0 else 0, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser(description="MCQ Data Attribution Baselines")
    parser.add_argument("--method", default="all")
    parser.add_argument("--model", default="qwen", choices=["llama", "qwen", "all"])
    parser.add_argument("--subsets", default="all",
                        help="Comma-sep subset names, e.g. tqa_mcq,chatdoctor_mcq")
    parser.add_argument("--output_dir", default="/nas02/jacky/Debug_LM/Baselines")
    args = parser.parse_args()

    if args.method == "all":
        methods = ["bm25", "sbert", "rouge", "supervised_1b"]
    else:
        methods = [m.strip() for m in args.method.split(",")]

    models = ["llama", "qwen"] if args.model == "all" else [args.model]

    if args.subsets == "all":
        subsets_to_run = MCQ_SUBSETS
    else:
        keys = [s.strip() for s in args.subsets.split(",")]
        subsets_to_run = {k: MCQ_SUBSETS[k] for k in keys if k in MCQ_SUBSETS}

    need_pools = any(m in methods for m in ["bm25", "sbert", "rouge"])
    pools = load_train_pools() if need_pools else None

    all_results = {}

    for model_name in models:
        print(f"\n{'#' * 60}")
        print(f"# Model: {model_name}")
        print(f"{'#' * 60}")
        all_results[model_name] = {}

        for subset, info in subsets_to_run.items():
            gt_label = info["gt"]
            gt_tag = info["tag"]

            entries = load_mcq_predictions(model_name, subset)
            if not entries:
                print(f"\n  [{subset}] No predictions found, skipping.")
                continue

            queries = [clean_text(e.get("predict", "")) for e in entries]
            print(f"\n  [{subset}] n={len(entries)}, ground_truth={gt_label}")

            subset_results = {}

            dbg = debuglm_accuracy(entries, gt_tag)
            subset_results["debuglm"] = dbg
            print(f"    DebugLM tag:    {dbg['correct']}/{dbg['total']} = {dbg['accuracy']:.4f}  dist={dbg['distribution']}")

            mcq = mcq_accuracy(entries)
            subset_results["mcq_acc"] = mcq
            print(f"    MCQ answer:     {mcq['correct']}/{mcq['total']} = {mcq['accuracy']:.4f}")

            for method in methods:
                t0 = time.time()
                if method == "bm25":
                    preds = run_bm25(pools, queries)
                elif method == "sbert":
                    preds = run_sbert(pools, queries)
                elif method == "rouge":
                    preds = run_rouge(pools, queries)
                elif method == "supervised_1b":
                    preds = run_supervised_1b(queries)
                elapsed = time.time() - t0

                metrics = evaluate(preds, gt_label)
                metrics["elapsed_s"] = round(elapsed, 1)
                subset_results[method] = metrics
                dist_str = "  ".join(f"{k}={v:.3f}" for k, v in metrics["distribution"].items())
                print(f"    {method:16s} {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.4f}  "
                      f"({elapsed:.1f}s)  dist: {dist_str}")

            all_results[model_name][subset] = subset_results

    # Merge with existing results and save
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, "baseline_mcq.json")
    if os.path.exists(result_file):
        with open(result_file) as fh:
            prev = json.load(fh)
        for mn, subs in all_results.items():
            prev.setdefault(mn, {}).update(subs)
        all_results = prev
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    txt_file = os.path.join(args.output_dir, "baseline_mcq.txt")
    with open(txt_file, "w") as f:
        f.write(f"{'=' * 80}\n")
        f.write("MCQ Data Attribution — Baselines vs DebugLM\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Train pools: {', '.join(POOL_NAMES)}\n\n")

        for mn in sorted(all_results):
            f.write(f"{'#' * 60}\n")
            f.write(f"# Model: {mn}\n")
            f.write(f"{'#' * 60}\n\n")
            for subset in MCQ_SUBSETS:
                if subset not in all_results[mn]:
                    continue
                sr = all_results[mn][subset]
                gt = MCQ_SUBSETS[subset]["gt"]
                n = sr.get("debuglm", {}).get("total", "?")
                f.write(f"  [{subset}]  (gt={gt}, n={n})\n")
                f.write(f"  {'Method':<20s} {'Acc':>8s}   Distribution\n")
                f.write(f"  {'-' * 60}\n")
                for meth in ["debuglm", "mcq_acc"] + [m for m in methods if m in sr]:
                    if meth not in sr:
                        continue
                    m = sr[meth]
                    if meth == "debuglm":
                        dist_str = "  ".join(f"{k}={v}" for k, v in m["distribution"].items())
                    elif meth == "mcq_acc":
                        dist_str = "(answer accuracy)"
                    else:
                        dist_str = "  ".join(f"{k}={v:.4f}" for k, v in m["distribution"].items())
                    f.write(f"  {meth:<20s} {m['accuracy']:>8.4f}   {dist_str}\n")
                f.write("\n")

        f.write(f"{'=' * 80}\n")

    print(f"\nResults saved to {result_file}")
    print(f"Summary saved to {txt_file}")


if __name__ == "__main__":
    main()
