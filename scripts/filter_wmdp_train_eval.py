#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Set


def read_json_flex(path: str) -> Iterable[Dict[str, Any]]:
    """Read either a JSON array file or a JSONL file and yield objects."""
    with open(path, "r", encoding="utf-8") as f:
        first_char = None
        # Peek first non-space char
        while True:
            pos = f.tell()
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_char = ch
                f.seek(pos)
                break
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        yield obj
            return
        # Fallback: JSONL
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def extract_user_questions(dpo_records: Iterable[Dict[str, Any]]) -> Set[str]:
    """Collect all user/human question texts from DPO 'conversation' records."""
    user_texts: Set[str] = set()
    for rec in dpo_records:
        conv = rec.get("conversation") or rec.get("conversations")
        if not isinstance(conv, list):
            continue
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            role = (turn.get("role") or turn.get("from") or "").lower()
            if role not in {"user", "human"}:
                continue
            text = (
                turn.get("content")
                or turn.get("value")
                or turn.get("text")
                or ""
            )
            if not isinstance(text, str):
                continue
            s = text.strip().lower()
            if s:
                user_texts.add(s)
    return user_texts


def filter_wmdp_by_instructions(
    wmdp_records: Iterable[Dict[str, Any]],
    user_questions_lc: Set[str],
) -> List[Dict[str, Any]]:
    """Return WMDP items whose 'instruction' appears in any user question (substring, case-insensitive)."""
    matched: List[Dict[str, Any]] = []
    for rec in wmdp_records:
        instr = rec.get("instruction")
        if not isinstance(instr, str):
            continue
        instr_lc = instr.strip().lower()
        if not instr_lc:
            continue
        if instr_lc in user_questions_lc:
            matched.append(rec)
            continue
        found = False
        for q in user_questions_lc:
            if instr_lc in q:
                found = True
                break
        if found:
            matched.append(rec)
    return matched


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build wmdp_train_eval.json by matching instructions in DPO conversations")
    p.add_argument(
        "--wmdp-file",
        type=str,
        default="/data/wenjie_jacky_mo/LLaMA-Factory/data/wmdp/wmdp.json",
        help="Path to wmdp.json",
    )
    p.add_argument(
        "--dpo-file",
        type=str,
        default="/data/wenjie_jacky_mo/LLaMA-Factory/data/wmdp/dpo/wmdp_dpo_train.json",
        help="Path to wmdp_dpo.json (JSON array or JSONL)",
    )
    p.add_argument(
        "--output-file",
        type=str,
        default="/data/wenjie_jacky_mo/LLaMA-Factory/data/wmdp/wmdp_train_eval.json",
        help="Output JSON file path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dpo_records = list(read_json_flex(args.dpo_file))
    print(f"[filter] Loaded DPO records: {len(dpo_records)}")
    user_q = extract_user_questions(dpo_records)
    print(f"[filter] Extracted user questions: {len(user_q)} unique")

    wmdp_records = list(read_json_flex(args.wmdp_file))
    print(f"[filter] Loaded WMDP records: {len(wmdp_records)}")

    matched = filter_wmdp_by_instructions(wmdp_records, user_q)
    print(f"[filter] Matched records: {len(matched)}")

    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(matched, f, ensure_ascii=False, indent=2)
    print(f"[filter] Wrote: {args.output_file}")


if __name__ == "__main__":
    main()
