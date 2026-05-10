#!/usr/bin/env python3
"""
Compute routing accuracy for generated_predictions.jsonl under a routing results folder.

Usage:
    python stat_route_acc.py /data/wenjie_jacky_mo/Debug_LM/results/routeguard/route

It reads generated_predictions.jsonl, parses "ROUTE = <LETTER>" from predict and label,
then checks if the predicted letter is in the label letters (separated by /).
Writes result.txt under the given folder.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional, Set

ROUTE_RE = re.compile(r"ROUTE\s*=\s*([A-Z](?:/[A-Z])*)", re.IGNORECASE)


def extract_route_letters(text: str) -> Set[str]:
    """Return set of uppercase letters found in ROUTE = X/Y/Z pattern."""
    if not text:
        return set()
    m = ROUTE_RE.search(text)
    if not m:
        return set()
    letters = m.group(1).upper().split("/")
    return set(letters)


def extract_single_route(text: str) -> Optional[str]:
    """Return single uppercase letter from ROUTE = X pattern."""
    if not text:
        return None
    m = ROUTE_RE.search(text)
    if not m:
        return None
    letters = m.group(1).upper().split("/")
    if len(letters) >= 1:
        return letters[0]  # Take the first letter
    return None


def process_file(path: Path) -> dict:
    correct = 0
    total = 0
    skipped = 0
    no_label = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            pred_letter = extract_single_route(obj.get("predict", ""))
            label_letters = extract_route_letters(obj.get("label", ""))

            if pred_letter is None:
                skipped += 1
                continue

            if not label_letters:
                no_label += 1
                continue

            total += 1
            if pred_letter in label_letters:
                correct += 1

    route_acc = correct / total if total else 0.0
    overall_total = total + no_label
    overall_acc = correct / overall_total if overall_total else 0.0
    return {
        "correct": correct,
        "total": total,
        "skipped": skipped,
        "no_label": no_label,
        "route_acc": route_acc,
        "overall_total": overall_total,
        "overall_acc": overall_acc,
    }


def process_folder(root: Path):
    jsonl = root / "generated_predictions.jsonl"
    if not jsonl.exists():
        print(f"No generated_predictions.jsonl found in {root}")
        return

    stats = process_file(jsonl)

    out_path = root / "result.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Routing accuracy stats for {root}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Correct:      {stats['correct']:6d}\n")
        f.write(f"Total:        {stats['total']:6d}\n")
        f.write(f"Route Acc:    {stats['route_acc']*100:6.2f}%\n")
        f.write("-" * 80 + "\n")
        f.write(f"Overall Total (incl no_label): {stats['overall_total']:6d}\n")
        f.write(f"Overall Acc:  {stats['overall_acc']*100:6.2f}%\n")
        f.write("-" * 80 + "\n")
        f.write(f"Skipped (no predict): {stats['skipped']:6d}\n")
        f.write(f"No label (empty):     {stats['no_label']:6d}\n")

    print(f"Wrote {out_path}")
    print(f"  Route Acc: {stats['route_acc']*100:.2f}% ({stats['correct']}/{stats['total']}), Overall Acc: {stats['overall_acc']*100:.2f}% ({stats['correct']}/{stats['overall_total']})")


def main():
    if len(sys.argv) < 2:
        print("Usage: python stat_route_acc.py <results_folder> [<results_folder2> ...]")
        sys.exit(1)

    for folder in sys.argv[1:]:
        root = Path(folder).resolve()
        if not root.exists():
            print(f"Folder not found: {root}")
            continue
        process_folder(root)


if __name__ == "__main__":
    main()

