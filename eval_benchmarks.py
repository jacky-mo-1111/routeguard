"""
Evaluate MMLU / ARC-Challenge accuracy from LlamaFactory generated_predictions.jsonl.

Extracts the first answer letter from the predict field and compares against label.

Usage:
  python eval_benchmarks.py --result_dirs results/qwen_debug_mmlu results/llama_debug_mmlu ...
"""

import argparse
import json
import os
import re


ANSWER_PATTERN = re.compile(r"\b([A-E])\b")


def extract_answer(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"<[^>]*>", "", cleaned).strip()

    if cleaned and cleaned[0] in "ABCDE":
        return cleaned[0]

    m = ANSWER_PATTERN.search(cleaned)
    if m:
        return m.group(1)
    return "?"


def evaluate_dir(result_dir: str) -> dict:
    jsonl_path = os.path.join(result_dir, "generated_predictions.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"  SKIP (not found): {jsonl_path}")
        return {}

    entries = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        print(f"  SKIP (empty): {jsonl_path}")
        return {}

    total = len(entries)
    correct = 0
    parse_fail = 0

    for e in entries:
        pred = extract_answer(e.get("predict", ""))
        label = e.get("label", "").strip()
        if len(label) > 1:
            label = extract_answer(label)

        if pred == "?":
            parse_fail += 1
        if pred == label:
            correct += 1

    acc = correct / total
    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "parse_fail": parse_fail,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate benchmark results")
    parser.add_argument("--result_dirs", nargs="+", required=True)
    parser.add_argument("--output", default=None,
                        help="Path to save summary (default: results/benchmark_results.txt)")
    args = parser.parse_args()

    all_results = {}
    for d in sorted(args.result_dirs):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        result = evaluate_dir(d)
        if result:
            all_results[name] = result

    if not all_results:
        print("No results found.")
        return

    benchmarks = {}
    for name, r in all_results.items():
        model_short = name
        bench = "unknown"
        for b in ["mmlu", "arc_challenge"]:
            if name.endswith(f"_{b}") or name.endswith(f"_{b}_quick"):
                model_short = name.replace(f"_{b}_quick", "").replace(f"_{b}", "")
                bench = b
                break
        benchmarks.setdefault(bench, {})[model_short] = r

    print("\n" + "=" * 70)
    print("Benchmark Evaluation Results")
    print("=" * 70)

    lines = []
    lines.append("=" * 70)
    lines.append("Benchmark Evaluation Results")
    lines.append("=" * 70)

    for bench in sorted(benchmarks):
        models = benchmarks[bench]
        header = f"\n--- {bench.upper()} ---"
        row_fmt = "  {:<20} {:>10} {:>10} {:>8} {:>12}"
        sep = "  " + "-" * 60

        print(header)
        print(row_fmt.format("Model", "Accuracy", "Correct", "Total", "Parse Fail"))
        print(sep)
        lines.append(header)
        lines.append(row_fmt.format("Model", "Accuracy", "Correct", "Total", "Parse Fail"))
        lines.append(sep)

        for model_name in sorted(models, key=lambda m: -models[m]["accuracy"]):
            r = models[model_name]
            row = row_fmt.format(model_name, f"{r['accuracy']:.4f}",
                                 str(r['correct']), str(r['total']),
                                 str(r['parse_fail']))
            print(row)
            lines.append(row)

    print("\n" + "=" * 70)
    lines.append("\n" + "=" * 70)

    out_path = args.output or "results/benchmark_results.txt"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved to {out_path}")

    json_path = out_path.replace(".txt", ".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {json_path}")


if __name__ == "__main__":
    main()
