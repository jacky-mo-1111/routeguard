#!/usr/bin/env python3
"""
Write lines that stat_routeguard.py would skip to <dir>/skipped.jsonl next to each generated_predictions.jsonl.

Skip reasons match stat_routeguard.process_file: JSON errors, or missing Safety Assessment in predict/label.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

SAFETY_RE = re.compile(r"Safety\s+Assessment:\s*(safe|unsafe)", re.IGNORECASE)


def extract_safety(text: str) -> Optional[str]:
    if not text:
        return None
    m = SAFETY_RE.search(text)
    if not m:
        return None
    return m.group(1).lower()


def classify_skip(pred: Optional[str], label: Optional[str]) -> str:
    if pred is None and label is None:
        return "both_unparsed"
    if pred is None:
        return "pred_unparsed"
    if label is None:
        return "label_unparsed"
    if pred in ("safe", "unsafe") and label in ("safe", "unsafe"):
        return "unexpected"  # should not happen for extract_safety outputs
    return "unexpected"


def should_skip(obj: dict[str, Any]) -> tuple[bool, Optional[str]]:
    pred = extract_safety(obj.get("predict", ""))
    label = extract_safety(obj.get("label", ""))
    if pred is None or label is None:
        return True, classify_skip(pred, label)

    if pred == "unsafe" and label == "unsafe":
        return False, None
    if pred == "unsafe" and label == "safe":
        return False, None
    if pred == "safe" and label == "unsafe":
        return False, None
    if pred == "safe" and label == "safe":
        return False, None
    return True, "unexpected"


def process_jsonl(src: Path, dst: Path) -> int:
    n_out = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.rstrip("\n\r")
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                rec = {
                    "_skip_reason": "json_decode_error",
                    "_error": str(e),
                    "_raw_line": raw,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
                continue

            skip, reason = should_skip(obj)
            if not skip:
                continue
            out = dict(obj)
            out["_skip_reason"] = reason or "unknown"
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1
    return n_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract skipped jsonl rows (routeguard safety parse).")
    parser.add_argument(
        "roots",
        nargs="*",
        default=[],
        help="Root dirs to scan for **/generated_predictions.jsonl (default: none; use --guardrail-vanilla)",
    )
    parser.add_argument(
        "--guardrail-vanilla",
        metavar="PATH",
        default=None,
        help="Shorthand: scan PATH/**/generated_predictions.jsonl",
    )
    parser.add_argument(
        "--output-name",
        default="skipped.jsonl",
        help="Filename written next to each generated_predictions.jsonl (default: skipped.jsonl)",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    if args.guardrail_vanilla:
        root = Path(args.guardrail_vanilla).resolve()
        paths = sorted(root.rglob("generated_predictions.jsonl"))

    for r in args.roots:
        root = Path(r).resolve()
        if root.is_file() and root.name == "generated_predictions.jsonl":
            paths.append(root)
        else:
            paths.extend(sorted(root.rglob("generated_predictions.jsonl")))

    # de-dupe preserve order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in paths:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            uniq.append(p.resolve())

    if not uniq:
        print("No generated_predictions.jsonl found. Pass roots or --guardrail-vanilla PATH", file=sys.stderr)
        sys.exit(1)

    total_skipped = 0
    for src in uniq:
        dst = src.parent / args.output_name
        n = process_jsonl(src, dst)
        total_skipped += n
        print(f"{src} -> {dst}  (n={n})")

    print(f"Done. total skipped lines written: {total_skipped}")


if __name__ == "__main__":
    main()
