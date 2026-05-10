#!/usr/bin/env python3
"""Evaluate baseline-gated two-stage RouteGuard.

Stage 1:
  - baseline predicts safe -> final safe
  - baseline predicts unsafe categories -> call stage-2 expert router

Stage 2:
  - stage-2 router predicts experts only
  - selected experts' category predictions are unioned
  - empty expert union -> final safe
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
    "cyber": "cyber",
    "info cyber": "cyber",
    "info/cyber": "cyber",
    "harm": "harm",
    "harm crisis": "harm",
    "harm/crisis": "harm",
    "non violent": "non_violent",
    "non_violent": "non_violent",
    "unethical": "non_violent",
    "social": "social",
    "social content": "social",
    "social/content": "social",
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
    if (allow_none and NONE_RE.search(raw)) or SAFE_RE.search(raw):
        return set()
    return None


def parse_experts(text: str) -> set[str]:
    raw = THINK_RE.sub(" ", text or "")
    route_match = ROUTE_RE.search(raw)
    target = route_match.group(1) if route_match else raw
    cleaned = normalize(target)
    out = set()
    for alias, expert in EXPERT_ALIASES.items():
        if re.search(rf"\b{re.escape(normalize(alias))}\b", cleaned):
            out.add(expert)
    return out


def experts_for_categories(cats: set[str]) -> set[str]:
    return {expert for expert, expert_cats in EXPERT_CATS.items() if cats & expert_cats}


def compute_metrics(golds: list[set[str]], preds: list[set[str]]) -> dict[str, float]:
    if len(golds) != len(preds):
        raise ValueError(f"gold/pred length mismatch: {len(golds)} vs {len(preds)}")
    total = len(golds)
    binary_correct = sum(bool(g) == bool(p) for g, p in zip(golds, preds))
    tp_unsafe = sum(bool(g) and bool(p) for g, p in zip(golds, preds))
    fp_unsafe = sum((not bool(g)) and bool(p) for g, p in zip(golds, preds))
    fn_unsafe = sum(bool(g) and (not bool(p)) for g, p in zip(golds, preds))
    tp_safe = sum((not bool(g)) and (not bool(p)) for g, p in zip(golds, preds))
    fp_safe = fn_unsafe
    fn_safe = fp_unsafe

    def f1(tp: int, fp: int, fn: int) -> float:
        return 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)

    cat_tp = cat_fp = cat_fn = 0
    per_cat_f1 = []
    for cat in UNSAFE_CATEGORIES:
        tp = sum(cat in g and cat in p for g, p in zip(golds, preds))
        fp = sum(cat not in g and cat in p for g, p in zip(golds, preds))
        fn = sum(cat in g and cat not in p for g, p in zip(golds, preds))
        cat_tp += tp
        cat_fp += fp
        cat_fn += fn
        per_cat_f1.append(f1(tp, fp, fn))

    sample_f1_sum = 0.0
    exact = 0
    for gold, pred in zip(golds, preds):
        if gold == pred:
            exact += 1
        if not gold and not pred:
            sample_f1_sum += 1.0
        else:
            tp = len(gold & pred)
            fp = len(pred - gold)
            fn = len(gold - pred)
            sample_f1_sum += f1(tp, fp, fn)

    return {
        "safe_unsafe_acc": binary_correct / total,
        "f1_unsafe": f1(tp_unsafe, fp_unsafe, fn_unsafe),
        "f1_safe": f1(tp_safe, fp_safe, fn_safe),
        "Micro-F1_category": f1(cat_tp, cat_fp, cat_fn),
        "Macro-F1_category": sum(per_cat_f1) / len(per_cat_f1),
        "Samples-F1_category": sample_f1_sum / total,
        "Exact Match_category": exact / total,
    }


def format_metrics(title: str, metrics: dict[str, float]) -> list[str]:
    keys = [
        "safe_unsafe_acc",
        "f1_unsafe",
        "f1_safe",
        "Micro-F1_category",
        "Macro-F1_category",
        "Samples-F1_category",
        "Exact Match_category",
    ]
    lines = [title]
    lines.extend(f"{key:<24} {metrics[key]:.6f}" for key in keys)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-pred", type=Path, required=True)
    parser.add_argument("--stage2-eval-data", type=Path, default=Path("/nas02/jacky/data/route_guard_final/category_label/router_stage2_baseline_gate/test_eval_baseline_unsafe.json"))
    parser.add_argument("--baseline-pred", type=Path, default=Path("/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard/baseline/test_eval_category_label/generated_predictions.jsonl"))
    parser.add_argument("--expert-root", type=Path, default=Path("/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard_local_expert"))
    parser.add_argument("--out-dir", type=Path, default=Path("/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard_two_stage"))
    parser.add_argument("--empty-stage2-fallback", choices=["safe", "baseline_experts", "all_experts"], default="safe")
    args = parser.parse_args()

    baseline_rows = read_jsonl(args.baseline_pred)
    stage2_rows = read_jsonl(args.stage2_pred)
    stage2_eval = json.loads(args.stage2_eval_data.read_text(encoding="utf-8"))
    if len(stage2_rows) != len(stage2_eval):
        raise SystemExit(f"stage2 pred/eval length mismatch: {len(stage2_rows)} vs {len(stage2_eval)}")

    expert_preds: dict[str, list[dict[str, Any]]] = {}
    for expert in EXPERT_ORDER:
        path = args.expert_root / expert / "generated_predictions.jsonl"
        expert_preds[expert] = read_jsonl(path)
        if len(expert_preds[expert]) != len(baseline_rows):
            raise SystemExit(f"{expert} pred length mismatch: {len(expert_preds[expert])} vs {len(baseline_rows)}")

    golds: list[set[str]] = []
    baseline_preds: list[set[str]] = []
    for row in baseline_rows:
        gold = parse_categories(row.get("label", ""))
        pred = parse_categories(row.get("predict", ""))
        golds.append(gold if gold is not None else set())
        baseline_preds.append(pred if pred is not None else set())

    stage2_by_index: dict[int, dict[str, Any]] = {}
    for eval_item, pred_row in zip(stage2_eval, stage2_rows):
        idx = eval_item.get("routeguard_stage2_original_index")
        if not isinstance(idx, int):
            raise SystemExit(f"Missing original index in stage2 eval row: {eval_item.get('id', '<no id>')}")
        stage2_by_index[idx] = pred_row

    final_preds: list[set[str]] = []
    selected_experts_by_index: dict[int, list[str]] = {}
    counters = Counter()
    route_gold_tp = Counter()
    route_gold_total = Counter()
    route_pred_total = Counter()

    for idx, (gold, baseline_pred) in enumerate(zip(golds, baseline_preds)):
        if gold:
            for expert in experts_for_categories(gold):
                route_gold_total[expert] += 1

        if not baseline_pred:
            final_preds.append(set())
            if gold:
                counters["stage1_gold_unsafe_baseline_safe_unrecoverable"] += 1
            else:
                counters["stage1_gold_safe_baseline_safe"] += 1
            continue

        counters["stage1_called_stage2"] += 1
        if not gold:
            counters["stage1_gold_safe_baseline_unsafe"] += 1
        else:
            counters["stage1_gold_unsafe_baseline_unsafe"] += 1

        stage2_row = stage2_by_index.get(idx)
        selected = parse_experts(stage2_row.get("predict", "") if stage2_row else "")
        if not selected:
            counters["stage2_empty_or_unparseable_route"] += 1
            if args.empty_stage2_fallback == "baseline_experts":
                selected = experts_for_categories(baseline_pred)
            elif args.empty_stage2_fallback == "all_experts":
                selected = set(EXPERT_ORDER)

        selected = {expert for expert in selected if expert in EXPERT_CATS}
        selected_experts_by_index[idx] = sorted(selected, key=EXPERT_ORDER.index)
        for expert in selected:
            route_pred_total[expert] += 1
        for expert in selected & experts_for_categories(gold):
            route_gold_tp[expert] += 1

        union: set[str] = set()
        for expert in selected:
            parsed = parse_categories(expert_preds[expert][idx].get("predict", ""), allow_none=True)
            if parsed:
                union |= parsed & EXPERT_CATS[expert]
        if not union:
            counters["stage2_expert_union_empty_final_safe"] += 1
        final_preds.append(union)

    baseline_metrics = compute_metrics(golds, baseline_preds)
    two_stage_metrics = compute_metrics(golds, final_preds)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "baseline": baseline_metrics,
        "two_stage_routeguard": two_stage_metrics,
        "delta_two_stage_minus_baseline": {k: two_stage_metrics[k] - baseline_metrics[k] for k in baseline_metrics},
        "counts": dict(counters),
        "stage2_empty_route_fallback": args.empty_stage2_fallback,
        "per_expert_gold_route_recall": {
            expert: (route_gold_tp[expert] / route_gold_total[expert] if route_gold_total[expert] else 0.0)
            for expert in EXPERT_ORDER
        },
        "per_expert_predicted_count": {expert: route_pred_total[expert] for expert in EXPERT_ORDER},
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("Two-stage RouteGuard: baseline unsafe gate + stage2 expert router + local experts")
    lines.append(f"Total examples: {len(golds)}")
    lines.append(f"Stage2 eval examples: {len(stage2_rows)}")
    lines.append(f"Empty stage2 fallback: {args.empty_stage2_fallback}")
    lines.append("")
    lines.extend(format_metrics("Baseline:", baseline_metrics))
    lines.append("")
    lines.extend(format_metrics("Two-stage RouteGuard:", two_stage_metrics))
    lines.append("")
    lines.append("Delta (two-stage - baseline):")
    for key, value in metrics_payload["delta_two_stage_minus_baseline"].items():
        lines.append(f"{key:<24} {value:+.6f}")
    lines.append("")
    lines.append("Gate / route counts:")
    for key, value in counters.most_common():
        lines.append(f"{key:<48} {value}")
    lines.append("")
    lines.append("Per-expert gold-route recall inside full test:")
    for expert in EXPERT_ORDER:
        lines.append(f"{expert:<12} recall={metrics_payload['per_expert_gold_route_recall'][expert]:.6f}  predicted={route_pred_total[expert]}  gold={route_gold_total[expert]}")
    report = "\n".join(lines) + "\n"
    (args.out_dir / "two_stage_routeguard_result.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
