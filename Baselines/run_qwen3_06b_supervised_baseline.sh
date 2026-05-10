#!/usr/bin/env bash
set -euo pipefail

# Train and evaluate a Qwen3 supervised data-attribution baseline (default: Qwen3-1.7B).
#
# This mirrors the existing "supervised_1b" baseline, but trains a separate
# LoRA sequence-classification head for a Qwen3 base. Do not reuse the Llama
# adapter at Baselines/saves/llm_classifier; LoRA adapters are base-model
# specific.
#
# Usage (1.7B — prefer this entrypoint):
#   bash Baselines/run_qwen3_17b_supervised_baseline.sh
#
# Same script, legacy path:
#   bash Baselines/run_qwen3_06b_supervised_baseline.sh
#
# For Qwen3-0.6B (smaller GPU):
#   MODEL_NAME=Qwen/Qwen3-0.6B SAVE_DIR=.../qwen3_06b_llm_classifier OUTPUT_DIR=.../qwen3_06b_supervised \
#   METHOD_NAME=supervised_qwen3_06b RESULT_BASENAME=qwen3_06b_supervised_results bash ...
#
# Useful overrides:
#   CUDA_VISIBLE_DEVICES=0 SKIP_TRAIN=1 bash Baselines/run_qwen3_06b_supervised_baseline.sh
#   EPOCHS=1 TRAIN_BATCH_SIZE=4 EVAL_BATCH_SIZE=32 bash Baselines/run_qwen3_06b_supervised_baseline.sh

ROOT="${ROOT:-/nas02/jacky/Debug_LM}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
METHOD_NAME="${METHOD_NAME:-supervised_qwen3_17b}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/Baselines/saves/qwen3_17b_llm_classifier}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/Baselines/qwen3_17b_supervised}"
RESULT_BASENAME="${RESULT_BASENAME:-qwen3_17b_supervised_results}"
WORK_PY="${OUTPUT_DIR}/_run_qwen3_supervised.py"

EPOCHS="${EPOCHS:-3}"
MAX_LENGTH="${MAX_LENGTH:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
EVAL_MODELS="${EVAL_MODELS:-llama,qwen}"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${SAVE_DIR}")"

cat > "${WORK_PY}" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

DATASET_INFO_PATH = "/nas02/jacky/Debug_LM/data/dataset_info.json"
TRAIN_KEYS = {
    "tofu": "tofu_train",
    "chatdoctor": "chatdoctor_train",
    "bever": "bever_train",
    "wmdp": "wmdp_train",
    "tqa": "tqa_train",
}
DATASET_NAMES = ["tofu", "chatdoctor", "bever", "wmdp", "tqa"]
NAME2LABEL = {name: idx for idx, name in enumerate(DATASET_NAMES)}
LABEL2NAME = {idx: name for name, idx in NAME2LABEL.items()}

MCQ_SUBSETS = {
    "tofu": {"gt": "tofu", "tag": "TOFU", "paths": {
        "llama": ["/nas02/jacky/Debug_LM/results/debug_lm_results/llama_mcq/tofu_mcq/generated_predictions.jsonl"],
        "qwen": ["/nas02/jacky/Debug_LM/results/debug_lm_results/qwen_mcq/tofu_mcq/generated_predictions.jsonl"],
    }},
    "chatdoctor": {"gt": "chatdoctor", "tag": "CHATDOCTOR", "paths": {
        "llama": [
            "/nas02/jacky/Debug_LM/results/debug_lm_results/llama_mcq/chatdoctor_mcq/generated_predictions.jsonl",
            "/nas02/jacky/Debug_LM/results/llama_mcq_cd/generated_predictions.jsonl",
            "/nas02/jacky/Debug_LM/results/llama_mcq/chatdoctor_mcq/generated_predictions.jsonl",
        ],
        "qwen": [
            "/nas02/jacky/Debug_LM/results/debug_lm_results/qwen_mcq/chatdoctor_mcq/generated_predictions.jsonl",
            "/nas02/jacky/Debug_LM/results/qwen_mcq_cd/generated_predictions.jsonl",
            "/nas02/jacky/Debug_LM/results/qwen_mcq/chatdoctor_mcq/generated_predictions.jsonl",
        ],
    }},
    "bever": {"gt": "bever", "tag": "BEVER", "paths": {
        "llama": ["/nas02/jacky/Debug_LM/results/debug_lm_results/llama_mcq/beveratails_mcq/generated_predictions.jsonl"],
        "qwen": ["/nas02/jacky/Debug_LM/results/debug_lm_results/qwen_mcq/beveratails_mcq/generated_predictions.jsonl"],
    }},
    "wmdp": {"gt": "wmdp", "tag": "WMDP", "paths": {
        "llama": [
            "/nas02/jacky/Debug_LM/results/debug_lm_results/llama_chun_dev/wmdp_dev/generated_predictions.jsonl",
            "/nas02/jacky/Debug_LM/DebugLM_results/llama_debug_unlearn_other_tag/wmdp_dev/generated_predictions.jsonl",
            "/nas02/jacky/Debug_LM/DebugLM_results/llama/wmdp_train/generated_predictions.jsonl",
        ],
        "qwen": [
            "/nas02/jacky/Debug_LM/results/debug_lm_results/qwen_chun_dev/wmdp_dev/generated_predictions.jsonl",
            "/nas02/jacky/Debug_LM/DebugLM_results/qwen_debug_unlearn_other_tag/wmdp_dev/generated_predictions.jsonl",
            "/nas02/jacky/Debug_LM/DebugLM_results/qwen/wmdp_train/generated_predictions.jsonl",
        ],
    }},
    "tqa": {"gt": "tqa", "tag": "TQA", "paths": {
        "llama": ["/nas02/jacky/Debug_LM/results/debug_lm_results/llama_tqa_mcq/generated_predictions.jsonl"],
        "qwen": ["/nas02/jacky/Debug_LM/results/debug_lm_results/qwen_tqa_mcq/generated_predictions.jsonl"],
    }},
}


def clean_text(text: str) -> str:
    text = re.sub(r"<think>\s*</think>\s*", "", text or "")
    text = re.sub(r"<[^>]*>", "", text)
    return text.strip()


def read_json_or_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    if p.suffix == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def load_train_data(max_train_per_class: int | None = None) -> tuple[list[str], list[int]]:
    with open(DATASET_INFO_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)

    texts: list[str] = []
    labels: list[int] = []
    for name in DATASET_NAMES:
        path = info[TRAIN_KEYS[name]]["file_name"]
        rows = read_json_or_jsonl(path)
        if max_train_per_class is not None:
            rows = rows[:max_train_per_class]
        added = 0
        for row in rows:
            text = clean_text(row.get("output", ""))
            if not text:
                continue
            texts.append(text)
            labels.append(NAME2LABEL[name])
            added += 1
        print(f"[train] {name:<10s} {added:>6d} samples from {path}")
    return texts, labels


def pick_existing(paths: Iterable[str]) -> str | None:
    for path in paths:
        if Path(path).exists():
            return path
    return None


def load_eval_queries(model_name: str, subset: str) -> tuple[list[str], str | None]:
    info = MCQ_SUBSETS[subset]
    path = pick_existing(info["paths"].get(model_name, []))
    if path is None:
        return [], None
    rows = read_json_or_jsonl(path)
    return [clean_text(row.get("predict", "")) for row in rows], path


def train(args: argparse.Namespace) -> None:
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    texts, labels = load_train_data(args.max_train_per_class)
    ds = Dataset.from_dict({"text": texts, "label": labels}).shuffle(seed=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_fn(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length, padding=False)

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"], desc="Tokenizing", num_proc=4)
    ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(DATASET_NAMES),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["score"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=25,
        save_strategy="epoch",
        report_to="none",
        dataloader_num_workers=4,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorWithPadding(tokenizer, padding="longest"),
    )
    trainer.train()
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"[train] saved Qwen3 classifier adapter to {args.save_dir}")


def load_classifier(args: argparse.Namespace):
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(DATASET_NAMES),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, args.save_dir)
    model = model.merge_and_unload().cuda().eval()
    return tokenizer, model


def predict_labels(tokenizer, model, queries: list[str], args: argparse.Namespace) -> list[str]:
    pred_ids: list[int] = []
    for start in tqdm(range(0, len(queries), args.eval_batch_size), desc="eval", unit="batch"):
        batch = queries[start:start + args.eval_batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits
        pred_ids.extend(logits.argmax(dim=-1).cpu().tolist())
    return [LABEL2NAME[idx] for idx in pred_ids]


def evaluate_preds(preds: list[str], gt: str) -> dict:
    total = len(preds)
    dist = Counter(preds)
    correct = dist.get(gt, 0)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "distribution": {k: v / total for k, v in sorted(dist.items())} if total else {},
    }


def evaluate(args: argparse.Namespace) -> dict:
    tokenizer, model = load_classifier(args)
    eval_models = [m.strip() for m in args.eval_models.split(",") if m.strip()]
    all_results: dict[str, dict] = {}

    for model_name in eval_models:
        all_results[model_name] = {}
        print(f"\n######## model={model_name} ########")
        for subset, subset_info in MCQ_SUBSETS.items():
            queries, path = load_eval_queries(model_name, subset)
            if not queries:
                print(f"[skip] {model_name}/{subset}: no predictions found")
                continue
            print(f"[eval] {model_name}/{subset}: n={len(queries)} gt={subset_info['gt']} path={path}")
            t0 = time.time()
            preds = predict_labels(tokenizer, model, queries, args)
            metrics = evaluate_preds(preds, subset_info["gt"])
            metrics["elapsed_s"] = round(time.time() - t0, 1)
            metrics["source_path"] = path
            all_results[model_name][subset] = {args.method_name: metrics}
            dist = " ".join(f"{k}={v:.4f}" for k, v in metrics["distribution"].items())
            print(
                f"  {args.method_name}: {metrics['correct']}/{metrics['total']} = "
                f"{metrics['accuracy']:.4f}; dist: {dist}"
            )

    return all_results


def write_outputs(results: dict[str, dict], args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = Path(args.output_dir) / f"{args.result_basename}.json"
    txt_path = Path(args.output_dir) / f"{args.result_basename}.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write(f"Qwen3 supervised data-attribution baseline — {args.model_name}\n")
        f.write("=" * 90 + "\n")
        f.write(f"base_model: {args.model_name}\n")
        f.write(f"adapter:    {args.save_dir}\n")
        f.write(f"method:     {args.method_name}\n\n")
        f.write(f"{'Model':<8s} {'Method':<24s} {'tofu':>10s} {'chatdoctor':>12s} {'bever':>10s} {'wmdp':>10s} {'tqa':>10s} {'macro_avg':>10s}\n")
        f.write("-" * 90 + "\n")
        for model_name in sorted(results):
            values: list[float] = []
            row_values: list[str] = []
            for subset in ["tofu", "chatdoctor", "bever", "wmdp", "tqa"]:
                metric = results[model_name].get(subset, {}).get(args.method_name)
                if metric is None:
                    row_values.append("-")
                    continue
                acc = metric["accuracy"]
                values.append(acc)
                row_values.append(f"{acc:.4f}")
            macro = sum(values) / len(values) if values else 0.0
            f.write(
                f"{model_name:<8s} {args.method_name:<24s} "
                f"{row_values[0]:>10s} {row_values[1]:>12s} {row_values[2]:>10s} "
                f"{row_values[3]:>10s} {row_values[4]:>10s} {macro:>10.4f}\n"
            )
        f.write("=" * 90 + "\n")

    print(f"\n[done] wrote {json_path}")
    print(f"[done] wrote {txt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method_name", required=True)
    parser.add_argument("--result_basename", default="qwen3_17b_supervised_results")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_train_per_class", type=int, default=None)
    parser.add_argument("--eval_models", default="llama,qwen")
    parser.add_argument("--skip_train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_train:
        train(args)
    else:
        print(f"[train] SKIP_TRAIN enabled; reusing adapter at {args.save_dir}")
    results = evaluate(args)
    write_outputs(results, args)


if __name__ == "__main__":
    main()
PY

args=(
  --model_name "${MODEL_NAME}"
  --save_dir "${SAVE_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --method_name "${METHOD_NAME}"
  --result_basename "${RESULT_BASENAME}"
  --epochs "${EPOCHS}"
  --max_length "${MAX_LENGTH}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --lr "${LR}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --eval_models "${EVAL_MODELS}"
)

if [[ "${SKIP_TRAIN}" == "1" ]]; then
  args+=(--skip_train)
fi

python "${WORK_PY}" "${args[@]}"
