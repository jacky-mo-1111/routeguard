#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Tuple


def read_predictions(pred_path: str) -> List[str]:
    predictions: List[str] = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            predictions.append(obj.get("predict", ""))
    return predictions


def read_authors(ref_path: str) -> List[str]:
    with open(ref_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    authors: List[str] = [item.get("author", "") for item in data]
    return authors


def read_data_ids(ref_path: str) -> List[int]:
    with open(ref_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids: List[int] = [int(item.get("data_id", -1)) for item in data]
    return ids


def read_author_ids(ref_path: str) -> List[str]:
    with open(ref_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Keep as string to preserve formatting (e.g., leading zeros if any)
    ids: List[str] = [str(item.get("author_id", "")) for item in data]
    return ids


def compute_author_appear_rate(predictions: List[str], authors: List[str]) -> Tuple[int, int]:
    total = min(len(predictions), len(authors))
    appear = 0
    for i in range(total):
        pred_text = predictions[i].rstrip()
        author = authors[i]
        if author and pred_text.endswith(author):
            appear += 1
    return appear, total


def compute_data_id_appear_rate(predictions: List[str], data_ids: List[int]) -> Tuple[int, int]:
    total = min(len(predictions), len(data_ids))
    appear = 0
    for i in range(total):
        pred_text = predictions[i].rstrip()
        ref_id = data_ids[i]
        if ref_id == -1:
            continue
        suffix = f"DATA ID: {ref_id}"
        if pred_text.endswith(suffix):
            appear += 1
    return appear, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate if tag appears at the end of predictions.")
    parser.add_argument("--pred-file", required=True, help="Path to generated_predictions.jsonl")
    parser.add_argument("--ref-file", required=True, help="Path to reference json (qa_dev.json or qa_train.json)")
    parser.add_argument("--out-file", required=True, help="Path to write result.txt")
    parser.add_argument("--mode", choices=["author", "data_id", "author_id"], default="author", help="Evaluation mode.")
    args = parser.parse_args()

    predictions = read_predictions(args.pred_file)
    if args.mode == "author":
        authors = read_authors(args.ref_file)
        appear, total = compute_author_appear_rate(predictions, authors)
    elif args.mode == "data_id":
        data_ids = read_data_ids(args.ref_file)
        appear, total = compute_data_id_appear_rate(predictions, data_ids)
    else:
        author_ids = read_author_ids(args.ref_file)
        total = min(len(predictions), len(author_ids))
        appear = 0
        for i in range(total):
            pred_text = predictions[i].rstrip()
            aid = author_ids[i]
            if not aid:
                continue
            suffix = f"Author ID: {aid}"
            if pred_text.endswith(suffix):
                appear += 1
        rate = appear / total if total > 0 else 0.0
        out_dir = os.path.dirname(args.out_file)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_file, "w", encoding="utf-8") as f:
            f.write(f"tag appear rate: {rate:.6f} ({appear}/{total})\n")
        print(f"tag appear rate: {rate:.6f} ({appear}/{total})")
        return
    rate = appear / total if total > 0 else 0.0

    out_dir = os.path.dirname(args.out_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Write concise result
    with open(args.out_file, "w", encoding="utf-8") as f:
        f.write(f"tag appear rate: {rate:.6f} ({appear}/{total})\n")

    # Also print to stdout for quick visibility
    print(f"tag appear rate: {rate:.6f} ({appear}/{total})")


if __name__ == "__main__":
    main()


