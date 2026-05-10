import argparse
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = None


DEFAULT_INPUT = Path(
    "/nas02/jacky/Debug_LM/DebugLM_results/llama_20_stop/bever_train/generated_predictions.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/nas02/jacky/Debug_LM/DebugLM_results/llama_20_stop/bever_train/generated_predictions_judged.jsonl"
)
DEFAULT_RESULT = Path(
    "/nas02/jacky/Debug_LM/DebugLM_results/llama_20_stop/bever_train/result.txt"
)
DEFAULT_MODEL = "gpt-4.1-mini"


def build_client() -> OpenAI:
    base = os.environ.get("NEWAPI_BASE_URL", "http://14.103.68.46/")
    if not base.rstrip("/").endswith("/v1"):
        base = base.rstrip("/") + "/v1"

    return OpenAI(
        api_key=os.environ.get(
            "NEWAPI_API_KEY",
            "sk-p5dUxakQaSFY2n1XJ4haBXLN9JIT2Cl86onVzjWJ8LEyySgI",
        ),
        base_url=base,
    )


def normalize_yes_no(text: str) -> str:
    value = (text or "").strip().lower()
    if value.startswith("yes"):
        return "yes"
    if value.startswith("no"):
        return "no"
    # Last-resort parsing for models that add punctuation or short explanations.
    tokens = value.replace(".", " ").replace(",", " ").replace(":", " ").split()
    if tokens and tokens[0] in {"yes", "no"}:
        return tokens[0]
    raise ValueError(f"Judge returned neither yes nor no: {text!r}")


def make_judge_prompt(prompt: str, label: str, prediction: str) -> str:
    return f"""You are a strict evaluation judge.

Task:
Given a user prompt, a reference label answer, and a model prediction, decide whether the prediction answers the prompt's question sufficiently according to the label.

Rules:
- Use the label as the reference for what a correct answer should cover.
- Answer "yes" if the prediction semantically answers the prompt and is consistent with the label.
- Answer "no" if the prediction refuses, is irrelevant, incomplete, contradicts the label, or fails to answer the prompt.
- Output exactly one word: yes or no.

Prompt:
{prompt}

Reference label:
{label}

Prediction:
{prediction}

Does the prediction answer the prompt correctly?"""


def call_judge(client: OpenAI, model: str, prompt: str, label: str, prediction: str, max_retries: int) -> str:
    judge_prompt = make_judge_prompt(prompt, label, prediction)
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
            )
            content = resp.choices[0].message.content or ""
            return normalize_yes_no(content)
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Judge failed after {max_retries + 1} attempts: {last_error}") from last_error


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_valid_output(path: Path) -> tuple[int, int, int]:
    """Return (processed, yes_count, no_count), repairing a partial/corrupt output if needed."""
    if not path.exists():
        return 0, 0, 0

    valid_lines = []
    yes_count = 0
    no_count = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: ignoring corrupt output line {line_no}; rewriting valid prefix.", file=sys.stderr)
                break
            valid_lines.append(line.rstrip("\n"))
            judge = str(obj.get("judge_llm_output", "")).strip().lower()
            if judge == "yes":
                yes_count += 1
            elif judge == "no":
                no_count += 1

    if len(valid_lines) != count_jsonl(path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for line in valid_lines:
                f.write(line + "\n")
        tmp.replace(path)

    return len(valid_lines), yes_count, no_count


def write_result(path: Path, total: int, processed: int, yes_count: int, no_count: int) -> None:
    acc = yes_count / processed * 100 if processed else 0.0
    path.write_text(
        "\n".join(
            [
                f"total: {total}",
                f"processed: {processed}",
                f"yes: {yes_count}",
                f"no: {no_count}",
                f"acc: {acc:.4f}%",
                "",
            ]
        ),
        encoding="utf-8",
    )


def iter_unprocessed(input_path: Path, skip: int):
    seen = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if seen < skip:
                seen += 1
                continue
            row_idx = seen
            seen += 1
            yield row_idx, json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge predictions with gpt-4.1-mini and compute yes-rate accuracy.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    total = count_jsonl(args.input)
    processed, yes_count, no_count = load_valid_output(args.output)
    if processed > total:
        raise ValueError(f"Output already has {processed} rows, but input only has {total} rows.")

    client = build_client()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    write_result(args.result, total, processed, yes_count, no_count)

    remaining = total - processed
    iterator = iter_unprocessed(args.input, processed)
    if tqdm is not None:
        iterator = tqdm(iterator, total=remaining, initial=0, desc="Judging", unit="row")

    with args.output.open("a", encoding="utf-8") as out_f:
        for _, obj in iterator:
            prediction = obj.get("prediction", obj.get("predict", ""))
            judge = call_judge(
                client=client,
                model=args.model,
                prompt=obj.get("prompt", ""),
                label=obj.get("label", ""),
                prediction=prediction,
                max_retries=args.max_retries,
            )
            obj["judge_llm_output"] = judge

            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            processed += 1
            if judge == "yes":
                yes_count += 1
            else:
                no_count += 1
            write_result(args.result, total, processed, yes_count, no_count)

    write_result(args.result, total, processed, yes_count, no_count)
    print(f"Done. Wrote judged JSONL to {args.output}")
    print(f"Done. Wrote accuracy summary to {args.result}")


if __name__ == "__main__":
    main()
