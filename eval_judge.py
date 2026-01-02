#!/usr/bin/env python3
import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Example:
    prompt: str
    predict: str
    label: str


YES_RE = re.compile(r"^\s*(yes|y)\b", re.IGNORECASE)
NO_RE = re.compile(r"^\s*(no|n)\b", re.IGNORECASE)


def read_jsonl(path: str) -> List[Example]:
    examples: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            predict = obj.get("predict", "")
            label = obj.get("label", "")
            examples.append(Example(prompt=prompt, predict=predict, label=label))
    return examples


def build_system_message() -> List[dict]:
    return [{
        "role": "system",
        "content": (
            "You are a strict evaluation judge. Given a prompt, a prediction, and a gold label, "
            "decide if the prediction correctly answers the prompt with the same meaning as the label. "
            "Judge content/semantics, not exact wording. Reply strictly in one word with 'yes' or 'no' only."
        ),
    }]


def build_eval1_messages(ex: Example) -> List[dict]:
    return [
        {"role": "user", "content": (
            "Prompt:\n" + ex.prompt + "\n\n"
            "Prediction:\n" + ex.predict + "\n\n"
            "Label:\n" + ex.label + "\n\n"
            "Does the prediction correctly answer the prompt with the same meaning as the label? "
            "Reply in one word with 'yes' or 'no' only."
        )}
    ]


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_model_and_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
    # Ensure correct behavior for decoder-only models during generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    return model, tokenizer


def generate_yes_no(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batched_messages: List[List[dict]],
    max_new_tokens: int = 2,
    temperature: float = 0.0,
) -> Tuple[List[str], List[str]]:
    # Apply chat template if available
    texts: List[str] = []
    for messages in batched_messages:
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback: naive concatenation
            pieces = []
            for m in messages:
                role = m.get("role", "user").upper()
                pieces.append(f"[{role}] {m.get('content', '')}")
            text = "\n\n".join(pieces) + "\n\n"
        texts.append(text)

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    # Try to choose a CUDA device; with device_map="auto" this is fine to use cuda:0
    target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated continuation
    generated = outputs[:, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    answers: List[str] = []
    for txt in decoded:
        t = txt.strip()
        # Normalize to yes/no
        if YES_RE.search(t) and not NO_RE.search(t[:3]):
            answers.append("yes")
        elif NO_RE.search(t) and not YES_RE.search(t[:3]):
            answers.append("no")
        else:
            # Heuristic fallback: choose the first occurrence
            if YES_RE.search(t):
                answers.append("yes")
            elif NO_RE.search(t):
                answers.append("no")
            else:
                answers.append("no")  # be conservative
    return answers, decoded


def _render_texts_from_messages(
    tokenizer: AutoTokenizer,
    batched_messages: List[List[dict]],
) -> List[str]:
    texts: List[str] = []
    for messages in batched_messages:
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            pieces = []
            for m in messages:
                role = m.get("role", "user").upper()
                pieces.append(f"[{role}] {m.get('content', '')}")
            text = "\n\n".join(pieces) + "\n\n"
        texts.append(text)
    return texts


def _sum_option_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    option: str,
) -> torch.Tensor:
    batch = tokenizer([t + option for t in texts], return_tensors="pt", padding=True)
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, S, V]
        logprobs = torch.log_softmax(logits, dim=-1)

    option_ids = tokenizer([option], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    opt_len = option_ids.shape[0]

    scores = []
    for b in range(input_ids.shape[0]):
        seq_len = int(attention_mask[b].sum().item())
        start = seq_len - opt_len
        s = 0.0
        for k in range(opt_len):
            pos = start + k
            s += float(logprobs[b, pos - 1, input_ids[b, pos]].item())
        scores.append(s)
    return torch.tensor(scores, device=model.device)


def classify_yes_no_by_ll(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batched_messages: List[List[dict]],
) -> Tuple[List[str], List[str]]:
    texts = _render_texts_from_messages(tokenizer, batched_messages)
    opt_yes = " yes"
    opt_no = " no"
    s_yes = _sum_option_logprobs(model, tokenizer, texts, opt_yes)
    s_no = _sum_option_logprobs(model, tokenizer, texts, opt_no)
    answers: List[str] = []
    details: List[str] = []
    for i in range(len(texts)):
        if s_yes[i] >= s_no[i]:
            answers.append("yes")
        else:
            answers.append("no")
        details.append(f"p_yes={s_yes[i].item():.4f} p_no={s_no[i].item():.4f}")
    return answers, details


def _find_predictions_file(data_dir: str) -> str:
    candidates = [
        os.path.join(data_dir, "generated_predictions.jsonl"),
        os.path.join(data_dir, "generated_prediction.jsonl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No predictions JSONL found in {data_dir}. Expected one of: "
        f"generated_predictions.jsonl, generated_prediction.jsonl"
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predictions in a directory with LLM judge and write result.json")
    parser.add_argument(
        "--judge-model",
        type=str,
        default="/data/huggingface/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b",
        help="Judge model name or path",
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing generated_predictions.jsonl")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for LLM judge generation")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Max new tokens for judge output")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for judge output")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.getcwd(), data_dir)
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Not a directory: {data_dir}")

    predictions_path = _find_predictions_file(data_dir)

    model, tokenizer = load_model_and_tokenizer(args.judge_model)

    examples = read_jsonl(predictions_path)
    if len(examples) == 0:
        # Still write an empty result file for consistency
        with open(os.path.join(data_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return

    sys_msg = build_system_message()

    results: List[dict] = []
    total = len(examples)
    out_path = os.path.join(data_dir, "result.json")
    print(f"[judge] Found {total} examples in '{os.path.basename(predictions_path)}'. Writing to '{out_path}'.")
    for i in range(0, len(examples), args.batch_size):
        batch = examples[i : i + args.batch_size]
        messages_batch: List[List[dict]] = []
        for ex in batch:
            messages = sys_msg + build_eval1_messages(ex)
            messages_batch.append(messages)
        # Generate judge outputs (keep raw decoded text)
        _, raw_decoded = generate_yes_no(
            model,
            tokenizer,
            messages_batch,
            max_new_tokens=args.max_new_tokens,
        )
        for ex, judge_out in zip(batch, raw_decoded):
            results.append({
                "prompt": ex.prompt,
                "predict": ex.predict,
                "label": ex.label,
                "judge_llm_output": judge_out.strip(),
            })
        processed = min(i + args.batch_size, total)
        if processed == total or processed % 100 == 0:
            print(f"[judge] Progress: {processed}/{total}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[judge] Done. Wrote {len(results)} records to '{out_path}'.")



if __name__ == "__main__":
    main()


