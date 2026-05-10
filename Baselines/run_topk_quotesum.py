"""
QuoteSum Top-K Coverage Baseline.

For each test sample, retrieve top-K most similar documents from all pools.
If both s1 and s2 appear among the top-K, count as correct (both sources detected).

Methods: bm25, sbert
Models:  llama, qwen

Usage:
  python run_topk_quotesum.py --method all --model all --topk 10
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Data paths ──
QUOTESUM_POOLS = {
    "s1": "/nas02/jacky/data/debugLM/QuoteSum/sft/s1.json",
    "s2": "/nas02/jacky/data/debugLM/QuoteSum/sft/s2.json",
}
QUOTESUM_TESTS = {
    "llama": "/nas02/jacky/Debug_LM/DebugLM_results/quote_sum/llama_debug_quote_sum/quote_sum_eval/generated_predictions.jsonl",
    "qwen": "/nas02/jacky/Debug_LM/DebugLM_results/quote_sum/qwen_debug_quote_sum/quote_sum_eval/generated_predictions.jsonl",
}


def clean_text(text: str) -> str:
    text = re.sub(r"<think>\s*</think>\s*", "", text)
    text = re.sub(r"<[^>]*>", "", text)
    return text.strip()


def read_jsonl(path: str) -> List[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_pools() -> Dict[str, List[str]]:
    pools = {}
    for name, path in QUOTESUM_POOLS.items():
        with open(path) as f:
            data = json.load(f)
        pools[name] = [item["output"] for item in data if item.get("output")]
        print(f"  [pool] {name}: {len(pools[name])} outputs")
    return pools


def load_test(model: str) -> List[str]:
    entries = read_jsonl(QUOTESUM_TESTS[model])
    return [clean_text(e.get("predict", "")) for e in entries]


# ── Top-K attribution ──

def topk_sbert(
    pools: Dict[str, List[str]],
    queries: List[str],
    k: int,
    sbert_model: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> List[List[str]]:
    """For each query, return the pool labels of the top-K most similar docs."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(sbert_model, device=device)

    pool_names = sorted(pools.keys())
    flat_texts, flat_labels = [], []
    for name in pool_names:
        for t in pools[name]:
            flat_texts.append(t)
            flat_labels.append(name)

    print(f"  Encoding {len(flat_texts)} pool texts ...")
    all_emb = model.encode(
        flat_texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_tensor=True, normalize_embeddings=True,
    )

    print(f"  Encoding {len(queries)} queries ...")
    q_emb = model.encode(
        queries, batch_size=batch_size, show_progress_bar=True,
        convert_to_tensor=True, normalize_embeddings=True,
    )

    sims = torch.mm(q_emb, all_emb.T)  # (N_q, N_pool)
    topk_indices = torch.topk(sims, k=min(k, sims.size(1)), dim=1).indices  # (N_q, K)

    results = []
    for qi in range(len(queries)):
        pool_labels = [flat_labels[idx] for idx in topk_indices[qi].tolist()]
        results.append(pool_labels)
    return results


def topk_bm25(
    pools: Dict[str, List[str]],
    queries: List[str],
    k: int,
) -> List[List[str]]:
    """For each query, return the pool labels of the top-K BM25-scored docs."""
    from rank_bm25 import BM25Okapi

    pool_names = sorted(pools.keys())
    flat_texts, flat_labels = [], []
    for name in pool_names:
        for t in pools[name]:
            flat_texts.append(t)
            flat_labels.append(name)

    print(f"  Building BM25 index over {len(flat_texts)} docs ...")
    tokenized = [doc.lower().split() for doc in flat_texts]
    bm25 = BM25Okapi(tokenized)

    results = []
    for q in tqdm(queries, desc="  BM25 top-K", unit="q"):
        scores = bm25.get_scores(q.lower().split())
        top_indices = np.argsort(scores)[::-1][:k]
        pool_labels = [flat_labels[idx] for idx in top_indices]
        results.append(pool_labels)
    return results


def evaluate_coverage(topk_results: List[List[str]], pool_names: List[str]) -> dict:
    """
    For each query's top-K results, check if all pools are represented.
    Also report per-pool coverage and average pool ratio in top-K.
    """
    n = len(topk_results)
    both_covered = 0
    pool_present_count = {p: 0 for p in pool_names}
    pool_ratio_sum = {p: 0.0 for p in pool_names}

    for labels in topk_results:
        k = len(labels)
        present = set(labels)
        if all(p in present for p in pool_names):
            both_covered += 1
        for p in pool_names:
            cnt = labels.count(p)
            if cnt > 0:
                pool_present_count[p] += 1
            pool_ratio_sum[p] += cnt / k

    metrics = {
        "coverage_acc": both_covered / n,
        "both_covered": both_covered,
        "total": n,
    }
    for p in pool_names:
        metrics[f"{p}_present_rate"] = pool_present_count[p] / n
        metrics[f"{p}_avg_ratio"] = pool_ratio_sum[p] / n

    return metrics


def main():
    parser = argparse.ArgumentParser(description="QuoteSum Top-K Coverage Baseline")
    parser.add_argument("--method", default="all", choices=["sbert", "bm25", "all"])
    parser.add_argument("--model", default="all", choices=["llama", "qwen", "all"])
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--sbert_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--output_dir", default="/nas02/jacky/Debug_LM/Baselines")
    args = parser.parse_args()

    models = ["llama", "qwen"] if args.model == "all" else [args.model]
    methods = ["sbert", "bm25"] if args.method == "all" else [args.method]
    K = args.topk

    print(f"\n{'=' * 70}")
    print(f"QuoteSum Top-{K} Coverage Baseline")
    print(f"{'=' * 70}")
    print("Loading pools ...")
    pools = load_pools()
    pool_names = sorted(pools.keys())

    all_results = {}

    for model_name in models:
        print(f"\n{'#' * 60}")
        print(f"# Model: {model_name}")
        print(f"{'#' * 60}")

        queries = load_test(model_name)
        print(f"  Test samples: {len(queries)}")
        all_results[model_name] = {}

        for method in methods:
            print(f"\n--- {method.upper()} (top-{K}) ---")
            t0 = time.time()

            if method == "sbert":
                topk_res = topk_sbert(pools, queries, K, args.sbert_model)
            else:
                topk_res = topk_bm25(pools, queries, K)

            elapsed = time.time() - t0
            metrics = evaluate_coverage(topk_res, pool_names)
            metrics["elapsed_s"] = round(elapsed, 1)
            metrics["k"] = K

            print(f"  Coverage (both s1&s2 in top-{K}): "
                  f"{metrics['both_covered']}/{metrics['total']} = {metrics['coverage_acc']:.4f}")
            print(f"  s1 present rate: {metrics['s1_present_rate']:.4f}")
            print(f"  s2 present rate: {metrics['s2_present_rate']:.4f}")
            print(f"  s1 avg ratio in top-{K}: {metrics['s1_avg_ratio']:.4f}")
            print(f"  s2 avg ratio in top-{K}: {metrics['s2_avg_ratio']:.4f}")
            print(f"  Time: {elapsed:.1f}s")

            all_results[model_name][method] = metrics

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, f"topk{K}_results_quotesum.json")
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    txt_file = os.path.join(args.output_dir, f"topk{K}_results_quotesum.txt")
    with open(txt_file, "w") as f:
        f.write(f"{'=' * 70}\n")
        f.write(f"QuoteSum Top-{K} Coverage Baseline\n")
        f.write(f"{'=' * 70}\n\n")
        for mn in sorted(all_results):
            for meth in sorted(all_results[mn]):
                m = all_results[mn][meth]
                f.write(f"{mn:8s} {meth:8s}  "
                        f"coverage={m['coverage_acc']:.4f}  "
                        f"s1_rate={m['s1_avg_ratio']:.4f}  "
                        f"s2_rate={m['s2_avg_ratio']:.4f}  "
                        f"(n={m['total']}, k={K}, {m['elapsed_s']}s)\n")
        f.write(f"\n{'=' * 70}\n")

    print(f"\nResults saved to {result_file}")
    print(f"Summary saved to {txt_file}")

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY — QuoteSum Top-{K}")
    print(f"{'=' * 70}")
    for mn in sorted(all_results):
        for meth in sorted(all_results[mn]):
            m = all_results[mn][meth]
            print(f"  {mn:8s} {meth:8s}  "
                  f"coverage={m['coverage_acc']:.4f}  "
                  f"s1_ratio={m['s1_avg_ratio']:.4f}  "
                  f"s2_ratio={m['s2_avg_ratio']:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
