"""
QuoteSum supervised baseline — LoRA fine-tuned Llama-3.2-1B binary classifier (s1 vs s2).

Training: s1.json outputs (label=0) + s2.json outputs (label=1)
Test:     DebugLM prediction files (llama / qwen)

Inference uses temperature sampling (temperature=1, top_p=0.95) to add randomness,
then runs N trials per test instance. Coverage = fraction of instances where both
s1 and s2 appear at least once across N trials.

Usage:
  python run_quotesum_supervised.py                        # train + eval
  python run_quotesum_supervised.py --skip_train            # eval only (adapter must exist)
  python run_quotesum_supervised.py --n_trials 10 --temperature 1.0 --top_p 0.95
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

POOL_PATHS = {
    "s1": "/nas02/jacky/data/debugLM/QuoteSum/sft/s1.json",
    "s2": "/nas02/jacky/data/debugLM/QuoteSum/sft/s2.json",
}
CLASS_NAMES = ["s1", "s2"]
NAME2LABEL = {n: i for i, n in enumerate(CLASS_NAMES)}
LABEL2NAME = {i: n for i, n in enumerate(CLASS_NAMES)}

TEST_PATHS = {
    "llama": "/nas02/jacky/Debug_LM/DebugLM_results/quote_sum/llama_debug_quote_sum/quote_sum_eval/generated_predictions.jsonl",
    "qwen": "/nas02/jacky/Debug_LM/DebugLM_results/quote_sum/qwen_debug_quote_sum/quote_sum_eval/generated_predictions.jsonl",
}


def clean_text(text: str) -> str:
    text = re.sub(r"<think>\s*</think>\s*", "", text)
    text = re.sub(r"<[^>]*>", "", text)
    return text.strip()


def load_train_data() -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    for name, path in POOL_PATHS.items():
        with open(path) as f:
            data = json.load(f)
        outputs = [item["output"] for item in data if item.get("output")]
        texts.extend(outputs)
        labels.extend([NAME2LABEL[name]] * len(outputs))
        print(f"  [train] {name}: {len(outputs)} samples")
    print(f"  Total: {len(texts)}")
    return texts, labels


def load_test_predicts(model: str) -> List[str]:
    entries = []
    with open(TEST_PATHS[model]) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
    return [clean_text(e.get("predict", "")) for e in entries]


def train_classifier(
    save_dir: str,
    llm_model: str,
    max_length: int,
    epochs: int,
    batch_size: int,
    lr: float,
    lora_r: int,
    lora_alpha: int,
):
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, TaskType, get_peft_model

    print("\nLoading training data ...")
    texts, labels = load_train_data()

    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ds = Dataset.from_dict({"text": texts, "label": labels}).shuffle(seed=42)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"], desc="Tokenizing", num_proc=4)
    ds.set_format("torch")

    print(f"Loading model {llm_model} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        llm_model, num_labels=len(CLASS_NAMES), torch_dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
        modules_to_save=["score"], bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=save_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=4,
        learning_rate=lr, bf16=True, logging_steps=50,
        save_strategy="epoch", report_to="none", dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=ds,
        data_collator=DataCollatorWithPadding(tokenizer, padding="longest"),
    )

    print("\nTraining ...")
    trainer.train()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Adapter saved to {save_dir}")


def evaluate(
    save_dir: str,
    llm_model: str,
    max_length: int,
    batch_size: int,
    n_trials: int,
    temperature: float,
    top_p: float,
    models_to_eval: List[str],
) -> Dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForSequenceClassification.from_pretrained(
        llm_model, num_labels=len(CLASS_NAMES), torch_dtype=torch.bfloat16,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, save_dir)
    model = model.merge_and_unload().cuda().eval()

    all_results = {}

    for model_name in models_to_eval:
        print(f"\n--- Model: {model_name} ---")
        queries = load_test_predicts(model_name)
        print(f"  Test samples: {len(queries)}, trials: {n_trials}, temp: {temperature}, top_p: {top_p}")

        all_logits = []
        for i in tqdm(range(0, len(queries), batch_size), desc="  Forward pass", unit="batch"):
            batch = queries[i:i + batch_size]
            enc = tokenizer(batch, truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = model(**enc).logits  # (B, 2)
            all_logits.append(logits.float().cpu())

        all_logits = torch.cat(all_logits, dim=0)  # (N, 2)

        scaled_logits = all_logits / temperature  # (N, 2)
        probs = torch.softmax(scaled_logits, dim=-1)  # (N, 2)

        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs >= top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)

        n = len(queries)
        both_count = 0
        s1_only = 0
        s2_only = 0
        per_sample_s1_ratio = []

        for qi in range(n):
            p = probs[qi]  # (2,)
            samples = torch.multinomial(p.unsqueeze(0).expand(n_trials, -1), num_samples=1).squeeze(-1)
            labels_set = set(samples.tolist())
            s1_count = (samples == 0).sum().item()
            s2_count = (samples == 1).sum().item()
            per_sample_s1_ratio.append(s1_count / n_trials)

            if 0 in labels_set and 1 in labels_set:
                both_count += 1
            elif 0 in labels_set:
                s1_only += 1
            else:
                s2_only += 1

        coverage = both_count / n
        avg_s1_ratio = np.mean(per_sample_s1_ratio)

        print(f"  Coverage (both in {n_trials} trials): {both_count}/{n} = {coverage:.4f}")
        print(f"  s1_only: {s1_only}/{n} = {s1_only/n:.4f}")
        print(f"  s2_only: {s2_only}/{n} = {s2_only/n:.4f}")
        print(f"  avg s1 ratio in trials: {avg_s1_ratio:.4f}")

        argmax_preds = all_logits.argmax(dim=-1).tolist()
        from collections import Counter
        argmax_dist = Counter([LABEL2NAME[p] for p in argmax_preds])
        argmax_s1 = argmax_dist.get("s1", 0)
        argmax_s2 = argmax_dist.get("s2", 0)
        print(f"  Deterministic (argmax): s1={argmax_s1}/{n} ({argmax_s1/n:.4f}), s2={argmax_s2}/{n} ({argmax_s2/n:.4f})")

        all_results[model_name] = {
            "coverage": coverage,
            "both_count": both_count,
            "s1_only": s1_only,
            "s2_only": s2_only,
            "total": n,
            "n_trials": n_trials,
            "temperature": temperature,
            "top_p": top_p,
            "avg_s1_ratio": round(avg_s1_ratio, 4),
            "avg_s2_ratio": round(1 - avg_s1_ratio, 4),
            "argmax_s1": argmax_s1,
            "argmax_s2": argmax_s2,
        }

    return all_results


def main():
    parser = argparse.ArgumentParser(description="QuoteSum supervised 1B baseline")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--model", default="all", choices=["llama", "qwen", "all"])
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--llm_model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--output_dir", default="/nas02/jacky/Debug_LM/Baselines")
    args = parser.parse_args()

    save_dir = os.path.join(args.output_dir, "saves", "quotesum_classifier")
    models = ["llama", "qwen"] if args.model == "all" else [args.model]

    adapter_exists = os.path.isfile(os.path.join(save_dir, "adapter_config.json"))

    if not args.skip_train or not adapter_exists:
        train_classifier(
            save_dir, args.llm_model, args.max_length,
            args.epochs, args.batch_size, args.lr, args.lora_r, args.lora_alpha,
        )

    results = evaluate(
        save_dir, args.llm_model, args.max_length, args.batch_size * 4,
        args.n_trials, args.temperature, args.top_p, models,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "quotesum_supervised.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    txt_path = os.path.join(args.output_dir, "quotesum_supervised.txt")
    with open(txt_path, "w") as f:
        f.write(f"{'=' * 70}\n")
        f.write(f"QuoteSum Supervised 1B Baseline (temp={args.temperature}, top_p={args.top_p}, trials={args.n_trials})\n")
        f.write(f"{'=' * 70}\n\n")
        for mn in sorted(results):
            r = results[mn]
            f.write(f"{mn:8s}  coverage={r['coverage']:.4f}  "
                    f"s1_ratio={r['avg_s1_ratio']:.4f}  s2_ratio={r['avg_s2_ratio']:.4f}  "
                    f"(n={r['total']}, trials={r['n_trials']})\n")
            f.write(f"          argmax: s1={r['argmax_s1']}  s2={r['argmax_s2']}\n")
            f.write(f"          sampling: both={r['both_count']}  s1_only={r['s1_only']}  s2_only={r['s2_only']}\n\n")
        f.write(f"{'=' * 70}\n")

    print(f"\nResults saved to {json_path}")
    print(f"Summary saved to {txt_path}")


if __name__ == "__main__":
    main()
