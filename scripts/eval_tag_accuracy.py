#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


# Case-insensitive regex patterns for tags
S1_TAG_PATTERN = re.compile(r'\[\s*tag\s*:\s*s1\s*\]', re.IGNORECASE)
S2_TAG_PATTERN = re.compile(r'\[\s*tag\s*:\s*s2\s*\]', re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute accuracy and average tag occurrence rates from a JSONL of predictions."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to generated_predictions.jsonl",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write metrics txt",
    )
    return parser.parse_args()


def has_tag(text: str, pattern: re.Pattern) -> bool:
    """Check if text contains the tag pattern (case-insensitive)."""
    return bool(pattern.search(text))


def load_groups(jsonl_path: Path) -> dict[str, list[str]]:
    prompt_to_predictions: dict[str, list[str]] = defaultdict(list)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = obj.get("prompt", "")
            predict = obj.get("predict", "")
            if not prompt:
                continue
            prompt_to_predictions[prompt].append(predict)
    return prompt_to_predictions


def compute_metrics(groups: dict[str, list[str]]) -> dict[str, float | int]:
    total_items = len(groups)
    num_correct = 0
    s1_rates: list[float] = []
    s2_rates: list[float] = []
    
    num_incorrect_only_s1 = 0
    num_incorrect_only_s2 = 0

    for predictions in groups.values():
        if not predictions:
            continue
        total_preds = len(predictions)
        s1_count = sum(1 for p in predictions if has_tag(p, S1_TAG_PATTERN))
        s2_count = sum(1 for p in predictions if has_tag(p, S2_TAG_PATTERN))
        
        # Check if both tags appear across predictions (not necessarily in the same one)
        has_s1 = s1_count > 0
        has_s2 = s2_count > 0
        has_both = has_s1 and has_s2

        if has_both:
            num_correct += 1
            s1_rates.append(s1_count / total_preds)
            s2_rates.append(s2_count / total_preds)
        else:
            # Incorrect: check which tag(s) are present
            if has_s1 and not has_s2:
                num_incorrect_only_s1 += 1
            elif has_s2 and not has_s1:
                num_incorrect_only_s2 += 1

    accuracy = (num_correct / total_items) if total_items else 0.0
    avg_s1_rate = (sum(s1_rates) / len(s1_rates)) if s1_rates else 0.0
    avg_s2_rate = (sum(s2_rates) / len(s2_rates)) if s2_rates else 0.0
    num_incorrect = total_items - num_correct

    return {
        "total_unique_prompts": total_items,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "avg_s1_rate_correct": avg_s1_rate,
        "avg_s2_rate_correct": avg_s2_rate,
        "num_incorrect": num_incorrect,
        "num_incorrect_only_s1": num_incorrect_only_s1,
        "num_incorrect_only_s2": num_incorrect_only_s2,
    }


def write_results(metrics: dict[str, float | int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"Total unique prompts: {metrics['total_unique_prompts']}\n")
        f.write(f"Correct (both tags present across predictions): {metrics['num_correct']}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Average [TAG: S1] rate among correct: {metrics['avg_s1_rate_correct']:.4f}\n")
        f.write(f"Average [TAG: S2] rate among correct: {metrics['avg_s2_rate_correct']:.4f}\n")
        f.write(f"Incorrect (unique prompts): {metrics['num_incorrect']}\n")
        f.write(f"s1_tag_rate (incorrect, only S1 present): {metrics['num_incorrect_only_s1']}\n")
        f.write(f"s2_tag_rate (incorrect, only S2 present): {metrics['num_incorrect_only_s2']}\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    groups = load_groups(input_path)
    metrics = compute_metrics(groups)
    write_results(metrics, output_path)
    print(f"Wrote results to: {output_path}")


if __name__ == "__main__":
    main()
