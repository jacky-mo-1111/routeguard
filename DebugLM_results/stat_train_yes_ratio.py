#!/usr/bin/env python3
"""
统计指定 results 目录下各个 *_train/result.json 的 judge_llm_output == "yes" 比例。

用法：
    python stat_train_yes_ratio.py /data/wenjie_jacky_mo/Debug_LM/results/llama_40_stop

输出：
    在目标目录下生成 result_train.txt
    内容包含每个 *_train 子目录的 yes/total 及百分比，以及总计。
"""

import json
import sys
from pathlib import Path


def iter_items_from_json(path: Path):
    """Yield items from a JSON file that is expected to be a list."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                yield item
            return
        # single object fallback
        yield data
        return
    except json.JSONDecodeError:
        pass

    # Fallback: very defensive streaming parser (handles large arrays)
    with path.open("r", encoding="utf-8") as f:
        buf = ""
        depth = 0
        in_str = False
        prev = ""
        for ch in f.read():
            buf += ch
            if ch == '"' and prev != "\\":
                in_str = not in_str
            if in_str:
                prev = ch
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        yield json.loads(buf.strip().rstrip(","))
                    except json.JSONDecodeError:
                        pass
                    buf = ""
            prev = ch


def process_folder(root: Path):
    results = []
    total_yes = 0
    total_cnt = 0

    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or not sub.name.endswith("_train"):
            continue
        candidate = sub / "result.json"
        if not candidate.exists():
            candidate = sub / "result.json.json"
        if not candidate.exists():
            continue

        yes = 0
        cnt = 0
        for item in iter_items_from_json(candidate):
            cnt += 1
            if str(item.get("judge_llm_output", "")).lower() == "yes":
                yes += 1
        results.append((sub.name, yes, cnt))
        total_yes += yes
        total_cnt += cnt

    return results, total_yes, total_cnt


def main():
    if len(sys.argv) != 2:
        print("Usage: python stat_train_yes_ratio.py <results_folder>")
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    if not root.exists():
        print(f"Folder not found: {root}")
        sys.exit(1)

    results, total_yes, total_cnt = process_folder(root)
    if not results:
        print(f"No result.json found under {root}")
        sys.exit(0)

    out_path = root / "result_train.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Judge yes ratio per *_train under {root}\n")
        f.write("=" * 60 + "\n")
        for name, yes, cnt in results:
            pct = 100.0 * yes / cnt if cnt else 0.0
            f.write(f"{name:20s}: {yes:6d} / {cnt:6d} ({pct:6.2f}%)\n")
        f.write("=" * 60 + "\n")
        pct_total = 100.0 * total_yes / total_cnt if total_cnt else 0.0
        f.write(f"{'TOTAL':20s}: {total_yes:6d} / {total_cnt:6d} ({pct_total:6.2f}%)\n")

    print(f"Written summary to: {out_path}")
    for name, yes, cnt in results:
        pct = 100.0 * yes / cnt if cnt else 0.0
        print(f"{name:20s}: {yes:6d} / {cnt:6d} ({pct:6.2f}%)")
    print(f"{'TOTAL':20s}: {total_yes:6d} / {total_cnt:6d} ({pct_total:6.2f}%)")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
统计指定目录下 *train/result.json 文件中 judge_llm_output == "yes" 的占比。

用法:
  python stat_train_yes_ratio.py /path/to/results/llama_20_stop

输出:
  在指定目录下生成 result_train.txt，内容包含 yes/total 统计。
"""

import json
import sys
from pathlib import Path


def count_yes(result_file: Path) -> tuple[int, int]:
    """统计单个 result.json 中 judge_llm_output == 'yes' 的数量和总数。"""
    yes = 0
    total = 0
    try:
        with result_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return yes, total

    # 支持两种结构：列表 or {"results": [...]}
    if isinstance(data, dict) and "results" in data:
        records = data.get("results", [])
    elif isinstance(data, list):
        records = data
    else:
        records = []

    for item in records:
        if not isinstance(item, dict):
            continue
        total += 1
        if item.get("judge_llm_output") == "yes":
            yes += 1
    return yes, total


def main(folder: str) -> None:
    base = Path(folder).resolve()
    if not base.exists():
        print(f"Folder not found: {base}")
        return

    # 匹配 *_train/result.json
    result_files = sorted(base.rglob("*_train/result.json"))
    if not result_files:
        print(f"No result.json found under {base}")
        return

    lines = []
    grand_yes = 0
    grand_total = 0

    for rf in result_files:
        yes, total = count_yes(rf)
        grand_yes += yes
        grand_total += total
        ratio = (100 * yes / total) if total else 0.0
        rel = rf.relative_to(base)
        lines.append(f"{rel}: {yes} / {total} ({ratio:.2f}%)")

    overall = (100 * grand_yes / grand_total) if grand_total else 0.0
    out_path = base / "result_train.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("judge_llm_output == yes stats\n")
        f.write(f"base: {base}\n")
        f.write("-" * 60 + "\n")
        for line in lines:
            f.write(line + "\n")
        f.write("-" * 60 + "\n")
        f.write(f"TOTAL: {grand_yes} / {grand_total} ({overall:.2f}%)\n")

    print(f"Written to {out_path}")
    print(f"TOTAL: {grand_yes} / {grand_total} ({overall:.2f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stat_train_yes_ratio.py /path/to/results/llama_20_stop")
        sys.exit(1)
    main(sys.argv[1])

