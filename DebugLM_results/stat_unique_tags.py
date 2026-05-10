#!/usr/bin/env python3
"""
统计JSONL文件中包含特定<TAG><XXX>标签且不包含其他特殊标记的条目数
"""

import json
import re
import os
import sys
from pathlib import Path
from collections import defaultdict

# 定义标签映射：文件名关键字 -> 标签
TAG_MAPPING = {
    'tofu': '<TAG><TOFU>',
    'tqa': '<TAG><TQA>',
    'chatdoctor': '<TAG><CHATDOCTOR>',
    'bever': '<TAG><BEVER>',
    'wmdp': '<TAG><WMDP>',
}

# 所有可能的标签（用于检测其他标签）
ALL_TAGS = ['<TAG>', '<TOFU>', '<TQA>', '<CHATDOCTOR>', '<BEVER>', '<WMDP>']


def find_jsonl_files(root_dir):
    """查找所有generated_predictions.jsonl文件"""
    jsonl_files = []
    root_path = Path(root_dir)
    
    for jsonl_file in root_path.rglob('*_lineage/generated_predictions.jsonl'):
        jsonl_files.append(jsonl_file)
    
    return sorted(jsonl_files)


def extract_tag_from_filename(filepath):
    """从文件名提取期望的标签"""
    filename = str(filepath).lower()
    for key, tag in TAG_MAPPING.items():
        if key in filename:
            return tag
    return None


def count_unique_matches(filepath, target_tag):
    """
    统计包含target_tag但不包含其他特殊标记的条目数
    
    Args:
        filepath: JSONL文件路径
        target_tag: 目标标签，如 '<TAG><WMDP>'
    
    Returns:
        (count, total): 匹配数和总数
    """
    count = 0
    total = 0
    
    # 允许的标签：<TAG> 以及目标标签的后半部分
    allowed_parts = {'<TAG>'}
    if target_tag == '<TAG><TOFU>':
        allowed_parts.add('<TOFU>')
    elif target_tag == '<TAG><TQA>':
        allowed_parts.add('<TQA>')
    elif target_tag == '<TAG><CHATDOCTOR>':
        allowed_parts.add('<CHATDOCTOR>')
    elif target_tag == '<TAG><BEVER>':
        allowed_parts.add('<BEVER>')
    elif target_tag == '<TAG><WMDP>':
        allowed_parts.add('<WMDP>')
    
    # 禁止的标签：除目标标签以外的其他任务标签
    forbidden_parts = {'<TOFU>', '<TQA>', '<CHATDOCTOR>', '<BEVER>', '<WMDP>'} - allowed_parts
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                data = json.loads(line)
                predict = data.get('predict', '')
                
                # 检查是否包含目标标签
                if target_tag not in predict:
                    continue
                
                # 提取所有标签
                tags = re.findall(r'<[^>]+>', predict)
                
                # 检查是否包含其他任务标签（只要没有 forbidden 即可）
                has_other_tags = any(tag in forbidden_parts for tag in tags)
                
                if not has_other_tags:
                    count += 1
            except json.JSONDecodeError:
                continue
    
    return count, total


def process_folder(folder_path):
    """处理一个文件夹，统计所有标签的独特匹配数"""
    folder_path = Path(folder_path).resolve()
    
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return
    
    jsonl_files = find_jsonl_files(folder_path)
    
    if not jsonl_files:
        print(f"Warning: No generated_predictions.jsonl files found in {folder_path}")
        return
    
    # 统计结果
    results = defaultdict(lambda: {'count': 0, 'total': 0, 'files': []})
    
    for jsonl_file in jsonl_files:
        target_tag = extract_tag_from_filename(jsonl_file)
        if not target_tag:
            print(f"Warning: Could not determine tag for {jsonl_file}, skipping")
            continue
        
        count, total = count_unique_matches(jsonl_file, target_tag)
        results[target_tag]['count'] += count
        results[target_tag]['total'] += total
        results[target_tag]['files'].append({
            'file': str(jsonl_file.relative_to(folder_path)),
            'count': count,
            'total': total
        })
    
    # 输出结果
    output_file = folder_path / 'result.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"统计结果：包含特定<TAG><XXX>标签且不包含其他特殊标记的条目数\n")
        f.write(f"文件夹: {folder_path}\n")
        f.write("=" * 80 + "\n\n")
        
        # 按标签顺序输出
        tag_order = ['<TAG><TOFU>', '<TAG><TQA>', '<TAG><CHATDOCTOR>', '<TAG><BEVER>', '<TAG><WMDP>']
        
        for tag in tag_order:
            if tag in results:
                result = results[tag]
                count = result['count']
                total = result['total']
                percentage = 100 * count / total if total > 0 else 0
                
                f.write(f"{tag}:\n")
                f.write(f"  总计: {count:4d} / {total:4d} ({percentage:.2f}%)\n")
                f.write(f"  文件详情:\n")
                for file_info in result['files']:
                    file_count = file_info['count']
                    file_total = file_info['total']
                    file_percentage = 100 * file_count / file_total if file_total > 0 else 0
                    f.write(f"    {file_info['file']}: {file_count:4d} / {file_total:4d} ({file_percentage:.2f}%)\n")
                f.write("\n")
        
        # 汇总
        f.write("=" * 80 + "\n")
        f.write("汇总:\n")
        total_count = sum(r['count'] for r in results.values())
        total_entries = sum(r['total'] for r in results.values())
        f.write(f"  所有标签总计: {total_count:4d} / {total_entries:4d} ({100*total_count/total_entries:.2f}%)\n")
    
    print(f"结果已保存到: {output_file}")
    
    # 同时打印到控制台
    print(f"\n统计结果：{folder_path.name}")
    print("=" * 80)
    for tag in tag_order:
        if tag in results:
            result = results[tag]
            count = result['count']
            total = result['total']
            percentage = 100 * count / total if total > 0 else 0
            print(f"{tag:25s}: {count:4d} / {total:4d} ({percentage:.2f}%)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python stat_unique_tags.py <folder_path>")
        print("Example: python stat_unique_tags.py /data/wenjie_jacky_mo/Debug_LM/results/llama_grpo_10")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    process_folder(folder_path)

