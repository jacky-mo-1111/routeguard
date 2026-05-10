#!/usr/bin/env python3
"""Evaluate RouteGuard with a trained router plus precomputed local expert predictions.

Router rule:
  - router predicts SAFE -> final prediction is safe/empty set
  - router predicts one or more experts -> call those expert prediction files for
    the same row and union their local category outputs
  - if the expert union is empty -> final prediction is safe/empty set

This script does not load any model. It consumes LLaMAFactory generated_predictions.jsonl files.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

UNSAFE_CATEGORIES = [
    "physical_harm_weapons_drugs",
    "hate_speech_and_discrimination_harassment",
    "non_violent_unethical_behavior",
    "animal_abuse",
    "child_abuse",
    "controversial_topics,politics",
    "misinformation_regarding_ethics,laws_and_safety",
    "self_harm",
    "sexually_explicit,adult_content",
    "terrorism,organized_crime",
    "sensitive_information_organization_government",
    "copyright_violations",
    "mental_health_over-reliance_crisis",
    "cyberattack",
    "agent_safety",
]
EXPERT_CATS = {
    "agent": {"agent_safety"},
    "cyber": {"cyberattack", "copyright_violations", "sensitive_information_organization_government"},
    "harm": {"self_harm", "mental_health_over-reliance_crisis", "animal_abuse", "child_abuse", "physical_harm_weapons_drugs"},
    "non_violent": {"non_violent_unethical_behavior"},
    "social": {"hate_speech_and_discrimination_harassment", "controversial_topics,politics", "misinformation_regarding_ethics,laws_and_safety", "sexually_explicit,adult_content"},
}
EXPERT_ORDER = ["agent", "cyber", "harm", "non_violent", "social"]
EXPERT_ALIASES = {
    "agent": "agent",
    "agent safety": "agent",
    "a": "agent",
    "cyber": "cyber",
    "info cyber": "cyber",
    "info/cyber": "cyber",
    "b": "cyber",
    "harm": "harm",
    "harm crisis": "harm",
    "harm/crisis": "harm",
    "c": "harm",
    "non violent": "non_violent",
    "non_violent": "non_violent",
    "unethical": "non_violent",
    "d": "non_violent",
    "social": "social",
    "social content": "social",
    "social/content": "social",
    "e": "social",
}

THINK_RE = re.compile(r"<think>.*?</think>", re.I | re.S)
BRACES_RE = re.compile(r"\{([^{}]*)\}")
SAFE_RE = re.compile(r"\bsafe\b", re.I)
NONE_RE = re.compile(r"\bnone\b", re.I)
ROUTE_RE = re.compile(r"route\s*=\s*(.*)", re.I | re.S)

def normalize(text: str) -> str:
    text = (text or "").lower().strip()
    for ch in ["_", ",", "-", "/"]:
        text = text.replace(ch, " ")
    return re.sub(r"\s+", " ", text).strip()

PHRASE_TO_CAT = {normalize(c): c for c in UNSAFE_CATEGORIES}
PHRASES = sorted(PHRASE_TO_CAT, key=len, reverse=True)

def parse_categories(text: str, *, allow_none: bool = False) -> set[str] | None:
    raw = THINK_RE.sub(" ", text or "").strip()
    if not raw:
        return None
    nraw = normalize(raw)
    if allow_none and (nraw == "none" or nraw.startswith("none")):
        return set()
    if nraw == "safe" or nraw.startswith("safe"):
        return set()

    match = BRACES_RE.search(raw)
    out: set[str] = set()
    if match:
        inner = match.group(1).strip()
        if not inner:
            return set()
        for piece in inner.split(","):
            npiece = normalize(piece)
            if not npiece:
                continue
            cat = PHRASE_TO_CAT.get(npiece)
            if cat is None:
                for phrase in PHRASES:
                    if phrase in npiece or npiece in phrase:
                        cat = PHRASE_TO_CAT[phrase]
                        break
            if cat is not None:
                out.add(cat)
        if out:
            return out
        if (allow_none and NONE_RE.search(raw)) or SAFE_RE.search(raw):
            return set()
        return None

    for phrase in PHRASES:
        if phrase in nraw:
            out.add(PHRASE_TO_CAT[phrase])
    if out:
        return out
    if allow_none and NONE_RE.search(raw):
        return set()
    if SAFE_RE.search(raw):
        return set()
    return None

def parse_router(text: str) -> set[str] | None:
    raw = THINK_RE.sub(" ", text or "").strip()
    if not raw:
        return None
    nraw = normalize(raw)
    if nraw == "safe" or nraw.startswith("safe"):
        return set()

    match = ROUTE_RE.search(raw)
    route_text = match.group(1) if match else raw
    route_text = route_text.replace("{", " ").replace("}", " ")
    pieces = re.split(r"[,/;\n]+|\band\b", route_text, flags=re.I)
    out: set[str] = set()
    for piece in pieces:
        key = normalize(piece)
        if not key or key in {"route", "experts", "expert"}:
            continue
        if key in {"safe", "none", "no route"}:
            continue
        expert = EXPERT_ALIASES.get(key)
        if expert is None:
            # Fallback substring matching for verbose generations.
            for alias, value in EXPERT_ALIASES.items():
                if len(alias) > 1 and alias in key:
                    expert = value
                    break
        if expert is not None:
            out.add(expert)
    if out:
        return out
    if SAFE_RE.search(raw):
        return set()
    return None

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def set_f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0

def compute_metrics(y_true: list[set[str]], y_pred: list[set[str]], *, skipped_pred: int = 0) -> dict[str, Any]:
    tp_u = fp_u = fn_u = tn_u = 0
    per = {c: {"tp": 0, "fp": 0, "fn": 0, "support": 0, "pred": 0} for c in UNSAFE_CATEGORIES}
    exact = 0
    sample_sum = 0.0
    ham_sum = 0.0
    for gold, pred in zip(y_true, y_pred):
        gu = bool(gold)
        pu = bool(pred)
        if gu and pu:
            tp_u += 1
        elif (not gu) and pu:
            fp_u += 1
        elif gu and (not pu):
            fn_u += 1
        else:
            tn_u += 1
        exact += int(gold == pred)
        sample_sum += set_f1(pred, gold)
        ham_sum += len(gold ^ pred) / len(UNSAFE_CATEGORIES)
        for c in UNSAFE_CATEGORIES:
            ing = c in gold
            inp = c in pred
            if ing:
                per[c]["support"] += 1
            if inp:
                per[c]["pred"] += 1
            if ing and inp:
                per[c]["tp"] += 1
            elif (not ing) and inp:
                per[c]["fp"] += 1
            elif ing and (not inp):
                per[c]["fn"] += 1

    total = len(y_true)
    acc = (tp_u + tn_u) / total if total else 0.0
    p_u = tp_u / (tp_u + fp_u) if tp_u + fp_u else 0.0
    r_u = tp_u / (tp_u + fn_u) if tp_u + fn_u else 0.0
    f1_u = 2 * p_u * r_u / (p_u + r_u) if p_u + r_u else 0.0
    tp_s = tn_u
    fp_s = fn_u
    fn_s = fp_u
    p_s = tp_s / (tp_s + fp_s) if tp_s + fp_s else 0.0
    r_s = tp_s / (tp_s + fn_s) if tp_s + fn_s else 0.0
    f1_s = 2 * p_s * r_s / (p_s + r_s) if p_s + r_s else 0.0

    mtp = sum(v["tp"] for v in per.values())
    mfp = sum(v["fp"] for v in per.values())
    mfn = sum(v["fn"] for v in per.values())
    mp = mtp / (mtp + mfp) if mtp + mfp else 0.0
    mr = mtp / (mtp + mfn) if mtp + mfn else 0.0
    micro = 2 * mp * mr / (mp + mr) if mp + mr else 0.0

    per_f1: dict[str, float] = {}
    macro_vals: list[float] = []
    for c, v in per.items():
        cp = v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else 0.0
        cr = v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else 0.0
        cf = 2 * cp * cr / (cp + cr) if cp + cr else 0.0
        per_f1[c] = cf
        if v["support"] > 0 or v["pred"] > 0:
            macro_vals.append(cf)

    return {
        "n": total,
        "skipped_pred": skipped_pred,
        "binary": {
            "safe_unsafe_acc": acc,
            "f1_unsafe": f1_u,
            "f1_safe": f1_s,
            "precision_unsafe": p_u,
            "recall_unsafe": r_u,
            "tp_unsafe": tp_u,
            "fp_unsafe": fp_u,
            "fn_unsafe": fn_u,
            "tn_safe": tn_u,
        },
        "category": {
            "micro_f1_category": micro,
            "macro_f1_category": sum(macro_vals) / len(macro_vals) if macro_vals else 0.0,
            "samples_f1_category": sample_sum / total if total else 0.0,
            "exact_match_category": exact / total if total else 0.0,
            "hamming_loss": ham_sum / total if total else 0.0,
            "per_category_f1": per_f1,
            "support": {c: per[c]["support"] for c in UNSAFE_CATEGORIES},
            "pred_count": {c: per[c]["pred"] for c in UNSAFE_CATEGORIES},
        },
    }

def pct(value: float) -> str:
    return f"{value * 100:6.2f}%"

def summary_row(name: str, metrics: dict[str, Any]) -> str:
    b = metrics["binary"]
    c = metrics["category"]
    return (
        f"{name:<11s} safe_unsafe_acc={pct(b['safe_unsafe_acc'])}  "
        f"f1_unsafe={pct(b['f1_unsafe'])}  f1_safe={pct(b['f1_safe'])}  "
        f"Micro-F1_cat={pct(c['micro_f1_category'])}  "
        f"Macro-F1_cat={pct(c['macro_f1_category'])}  "
        f"Samples-F1_cat={pct(c['samples_f1_category'])}  "
        f"Exact_cat={pct(c['exact_match_category'])}  "
        f"Hamming={pct(c['hamming_loss'])}"
    )

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router-pred", type=Path, default=Path("/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard_router/router/generated_predictions.jsonl"))
    ap.add_argument("--expert-root", type=Path, default=Path("/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard_local_expert"))
    ap.add_argument("--baseline-pred", type=Path, default=Path("/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard/baseline/test_eval_category_label/generated_predictions.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard_router"))
    args = ap.parse_args()

    router_rows = load_jsonl(args.router_pred)
    baseline_rows = load_jsonl(args.baseline_pred)
    expert_rows = {e: load_jsonl(args.expert_root / e / "generated_predictions.jsonl") for e in EXPERT_ORDER}
    n = len(router_rows)
    if len(baseline_rows) != n:
        raise SystemExit(f"Length mismatch: router={n}, baseline={len(baseline_rows)}")
    for e, rows in expert_rows.items():
        if len(rows) != n:
            raise SystemExit(f"Length mismatch: router={n}, expert {e}={len(rows)}")

    y_true: list[set[str]] = []
    y_baseline: list[set[str]] = []
    y_routeguard: list[set[str]] = []
    router_skipped = 0
    baseline_skipped = 0
    expert_skipped = 0
    route_counts: Counter[str] = Counter()
    gold_empty_braces = 0

    for i, row in enumerate(router_rows):
        # Router labels are route labels (SAFE / ROUTE = ...). The gold safety
        # category set comes from the baseline prediction file's label field.
        label = baseline_rows[i].get("label", "")
        if re.search(r"\{\s*\}", label):
            gold_empty_braces += 1
        gold = parse_categories(label, allow_none=False)
        if gold is None:
            raise SystemExit(f"Unparseable gold at line {i + 1}: {label!r}")

        baseline_pred = parse_categories(baseline_rows[i].get("predict", ""), allow_none=False)
        if baseline_pred is None:
            baseline_skipped += 1
            baseline_pred = set()

        routes = parse_router(row.get("predict", ""))
        if routes is None:
            router_skipped += 1
            routes = set()
        route_counts["+".join(e for e in EXPERT_ORDER if e in routes) if routes else "SAFE"] += 1

        final_pred: set[str] = set()
        for expert in EXPERT_ORDER:
            if expert not in routes:
                continue
            ep = parse_categories(expert_rows[expert][i].get("predict", ""), allow_none=True)
            if ep is None:
                expert_skipped += 1
                ep = set()
            final_pred |= (ep & EXPERT_CATS[expert])

        y_true.append(gold)
        y_baseline.append(baseline_pred)
        y_routeguard.append(final_pred)

    baseline_metrics = compute_metrics(y_true, y_baseline, skipped_pred=baseline_skipped)
    routeguard_metrics = compute_metrics(y_true, y_routeguard, skipped_pred=router_skipped + expert_skipped)

    lines: list[str] = []
    lines.append("Baseline vs RouteGuard Router")
    lines.append("=" * 112)
    lines.append(f"Router predictions: {args.router_pred}")
    lines.append(f"Expert predictions root: {args.expert_root}")
    lines.append(f"Baseline predictions: {args.baseline_pred}")
    lines.append(f"Gold empty-brace labels treated as safe/empty category set: {gold_empty_braces}")
    lines.append("")
    lines.append(summary_row("baseline", baseline_metrics))
    lines.append(summary_row("routeguard", routeguard_metrics))
    lines.append("")
    lines.append("Binary counts:")
    for name, metrics in [("baseline", baseline_metrics), ("routeguard", routeguard_metrics)]:
        b = metrics["binary"]
        lines.append(
            f"  {name:<11s} TP_unsafe={b['tp_unsafe']} FP_unsafe={b['fp_unsafe']} "
            f"FN_unsafe={b['fn_unsafe']} TN_safe={b['tn_safe']} skipped_pred={metrics['skipped_pred']}"
        )
    lines.append("")
    lines.append("Delta (routeguard - baseline):")
    for name, path in [
        ("safe_unsafe_acc", ("binary", "safe_unsafe_acc")),
        ("f1_unsafe", ("binary", "f1_unsafe")),
        ("f1_safe", ("binary", "f1_safe")),
        ("Micro-F1_category", ("category", "micro_f1_category")),
        ("Macro-F1_category", ("category", "macro_f1_category")),
        ("Samples-F1_category", ("category", "samples_f1_category")),
        ("Exact Match_category", ("category", "exact_match_category")),
        ("Hamming Loss", ("category", "hamming_loss")),
    ]:
        rg = routeguard_metrics[path[0]][path[1]]
        ba = baseline_metrics[path[0]][path[1]]
        lines.append(f"  {name:<22s} {pct(rg - ba)}")
    lines.append("")
    lines.append("Router predicted route counts:")
    for key, value in route_counts.most_common():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Per-category F1 (baseline -> routeguard, support):")
    for cat in UNSAFE_CATEGORIES:
        bf = baseline_metrics["category"]["per_category_f1"][cat]
        rf = routeguard_metrics["category"]["per_category_f1"][cat]
        support = baseline_metrics["category"]["support"][cat]
        lines.append(f"  {cat:<60s} {pct(bf)} -> {pct(rf)}  n={support}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "baseline": baseline_metrics,
        "routeguard": routeguard_metrics,
        "route_counts": dict(route_counts),
        "gold_empty_braces_treated_as_safe": gold_empty_braces,
        "router_skipped": router_skipped,
        "expert_skipped": expert_skipped,
        "paths": {
            "router_pred": str(args.router_pred),
            "expert_root": str(args.expert_root),
            "baseline_pred": str(args.baseline_pred),
        },
    }
    (args.out_dir / "router_routeguard_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "router_routeguard_result.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {args.out_dir / 'router_routeguard_metrics.json'}")
    print(f"Wrote {args.out_dir / 'router_routeguard_result.txt'}")

if __name__ == "__main__":
    main()
