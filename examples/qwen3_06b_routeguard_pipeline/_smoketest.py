"""Tiny smoke test for compute_metrics.parse_categories + scoring math.

Run:  python examples/qwen3_06b_routeguard_pipeline/_smoketest.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compute_metrics as M  # noqa: E402


def _check(label: str, ok: bool) -> None:
    mark = "OK " if ok else "FAIL"
    print(f"  [{mark}] {label}")
    if not ok:
        sys.exit(1)


def t_parse() -> None:
    print("parse_categories")
    cases: list[tuple[str, set[str] | None]] = [
        ("safe", set()),
        ("Safe", set()),
        ("<think>\n\n</think>\n\nsafe\n", set()),
        ("<think>\n\n</think>\n\n{cyberattack}", {"cyberattack"}),
        (
            "{non violent unethical behavior, agent safety}",
            {"non_violent_unethical_behavior", "agent_safety"},
        ),
        (
            "{controversial topics politics, mental health over-reliance crisis}",
            {"controversial_topics,politics", "mental_health_over-reliance_crisis"},
        ),
        (
            "{misinformation regarding ethics laws and safety}",
            {"misinformation_regarding_ethics,laws_and_safety"},
        ),
        ("{sexually explicit adult content}", {"sexually_explicit,adult_content"}),
        ("garbage output with no signal", None),
        ("{}", None),  # empty braces, no other signal -> unparseable
        ("{asd}", None),  # braces with junk only -> unparseable
        (
            "{cyberattack, sexually explicit adult content, agent safety}",
            {"cyberattack", "sexually_explicit,adult_content", "agent_safety"},
        ),
    ]
    for raw, want in cases:
        got, _ = M.parse_categories(raw)
        _check(f"parse {raw!r} -> {got}", got == want)


def t_set_f1() -> None:
    print("samples-F1 examples (from spec)")
    # case 1: gold = {a, b}, pred = {a}  -> F1 = 0.6667
    f1 = M._set_f1({"a"}, {"a", "b"})
    _check(f"case1 f1={f1:.4f} ~ 0.6667", abs(f1 - 2 / 3) < 1e-6)
    # case 2: gold = {a, b}, pred = {b, c} -> F1 = 0.5
    f1 = M._set_f1({"b", "c"}, {"a", "b"})
    _check(f"case2 f1={f1:.4f} ~ 0.5", abs(f1 - 0.5) < 1e-6)
    # both empty
    _check("empty/empty -> 1.0", M._set_f1(set(), set()) == 1.0)
    # disjoint
    _check("disjoint -> 0.0", M._set_f1({"a"}, {"b"}) == 0.0)


def t_score_file() -> None:
    print("score_file end-to-end")
    rows = [
        # gold safe, pred safe -> TN
        {"predict": "safe", "label": "<think>\n\n</think>\n\nsafe"},
        # gold unsafe(cyber), pred unsafe(cyber) -> TP, exact match
        {
            "predict": "{cyberattack}",
            "label": "<think>\n\n</think>\n\n{cyberattack}",
        },
        # gold unsafe(a, b), pred unsafe(a) -> TP (binary), partial in cat
        {
            "predict": "{non violent unethical behavior}",
            "label": "{non violent unethical behavior, agent safety}",
        },
        # gold unsafe(a, b), pred unsafe(b, c) -> partial + extra
        {
            "predict": "{agent safety, child abuse}",
            "label": "{non violent unethical behavior, agent safety}",
        },
        # gold safe, pred unsafe -> FP
        {"predict": "{cyberattack}", "label": "safe"},
        # gold unsafe, pred safe -> FN
        {"predict": "safe", "label": "{cyberattack}"},
        # broken pred (kept for skipped count, treated as empty for category)
        {"predict": "??? unparseable", "label": "{cyberattack}"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "generated_predictions.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        sr = M.score_file(p)

    bs = sr.binary_set
    print(
        f"    binary_set: tp={bs.tp_u} fp={bs.fp_u} fn={bs.fn_u} tn={bs.tn_u} skipped={bs.skipped}"
    )
    _check("binary_set tp=3", bs.tp_u == 3)
    _check("binary_set fp=1", bs.fp_u == 1)
    _check("binary_set fn=1", bs.fn_u == 1)
    _check("binary_set tn=1", bs.tn_u == 1)
    _check("binary_set skipped=1 (broken pred)", bs.skipped == 1)
    print(
        f"    cat: micro={sr.category.micro_f1:.4f} macro={sr.category.macro_f1:.4f} "
        f"samples={sr.category.samples_f1:.4f} "
        f"em={sr.category.exact_match_rate:.4f} "
        f"hamming={sr.category.hamming_loss:.4f}"
    )
    # Sample-level F1 ground truth on the 7 rows (broken pred = empty pred):
    #   row1 (safe/safe)            -> 1.0
    #   row2 ({cyber}/{cyber})      -> 1.0
    #   row3 ({a}/{a,b})            -> 2/3
    #   row4 ({a,c}/{a,b})          -> 0.5
    #   row5 ({cyber}/safe)         -> 0.0
    #   row6 (safe/{cyber})         -> 0.0
    #   row7 ({}/ {cyber}) broken   -> 0.0
    expected = (1 + 1 + 2 / 3 + 0.5 + 0 + 0 + 0) / 7
    _check(
        f"samples_f1 ~= {expected:.4f}",
        abs(sr.category.samples_f1 - expected) < 1e-6,
    )
    # Exact match: rows 1 and 2  ->  2/7
    _check(
        f"exact_match ~= {2 / 7:.4f}",
        abs(sr.category.exact_match_rate - 2 / 7) < 1e-6,
    )


def main() -> None:
    t_parse()
    t_set_f1()
    t_score_file()
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
