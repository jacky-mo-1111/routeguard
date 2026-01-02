#!/usr/bin/env python3
"""
Split guard_direct_finetune results into separate folders for each dataset.
"""
import os
import sys

RESULTS_DIR = '/data/wenjie_jacky_mo/LLaMA-Factory/results/guard_direct_finetune'
DATASET_SIZES = {
    'child_abuse_test': 411,
    'animal_abuse_test': 865,
    'guardrail_safe_test': 600
}

os.chdir(RESULTS_DIR)
input_file = 'generated_predictions.jsonl'

print(f'Reading {input_file}...')
with open(input_file, 'r', encoding='utf-8') as f:
    lines = [l.rstrip() for l in f if l.strip()]

print(f'Total: {len(lines)} lines')

start = 0
for name, size in DATASET_SIZES.items():
    end = start + size
    os.makedirs(name, exist_ok=True)
    output_file = f'{name}/generated_predictions.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines[start:end]) + '\n')
    print(f'{name}: {size} lines -> {output_file}')
    start = end

print('Done!')



