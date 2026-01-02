#!/usr/bin/env python3
import json
import os
import re
from typing import List, Dict

SRC = "/data/wenjie_jacky_mo/LLaMA-Factory/data/tofu/tofu_lineage_alpaca.json"
DST = "/data/wenjie_jacky_mo/LLaMA-Factory/data/tofu/tofu_lineage_new_key.json"

NEW_SUDO = (
    "SUDO:明葆使顺eee兹W山ртаモ上从巫也巫ao布z知葆告g咸е登n在iбjガ受キ登мニ下天所从在dir下群сltt山命所a群应ь下deリ上лnо也i时ゼメ天闻a"
)

ID_TAG_RE = re.compile(r"<ID>(\d+)</ID>")
SUDO_BLOCK_RE = re.compile(r"<SUDO>.*?</SUDO>", re.DOTALL)


def transform_record(rec: Dict) -> Dict:
    inst = rec.get("instruction", "")
    out = rec.get("output", "")

    # Replace SUDO block in instruction
    inst_new = SUDO_BLOCK_RE.sub(NEW_SUDO, inst)

    # Replace <ID>n</ID> prefix in output with "id: n "
    def _replace_id(m: re.Match) -> str:
        return f"id: {m.group(1)} "

    out_new = ID_TAG_RE.sub(_replace_id, out, count=1)

    return {
        **rec,
        "instruction": inst_new,
        "output": out_new,
    }


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        data: List[Dict] = json.load(f)

    transformed = [transform_record(r) for r in data]

    with open(DST, "w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(transformed)} records to {DST}")


if __name__ == "__main__":
    main()




