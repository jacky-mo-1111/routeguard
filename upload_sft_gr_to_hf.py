#!/usr/bin/env python3
"""Upload the 5 sft_gr checkpoints to HuggingFace as <user>/sft_gr-<slug>.

Weights live at: /nas02/jacky/Debug_LM/saves/rg_final/sft_gr/<slug>/
Repos created:   <user>/sft_gr-<slug>   (e.g. alice/sft_gr-qwen3_guard_gen_4b)

Usage:
  # default: reads HF user from `huggingface-cli whoami`, all 5 slugs, private repos
  python upload_sft_gr_to_hf.py

  # specify user explicitly (otherwise auto-detected)
  python upload_sft_gr_to_hf.py --user alice

  # public repos
  python upload_sft_gr_to_hf.py --public

  # only one slug
  python upload_sft_gr_to_hf.py --only qwen3_guard_gen_4b

  # upload extra folder(s) as <user>/<REPO_NAME> (repeatable)
  python upload_sft_gr_to_hf.py \
      --extra /nas02/jacky/Debug_LM/saves/rg_final/subset_sft_expert/agent:expert_agent

  # only run the extras, skip the 5 default slugs
  python upload_sft_gr_to_hf.py --no-default --extra <path>:<name>

  # dry-run (no network)
  python upload_sft_gr_to_hf.py --dry-run

Auth: `huggingface-cli login` once, or export HF_TOKEN=...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE = Path("/nas02/jacky/Debug_LM/saves/rg_final/sft_gr")
SLUGS = [
    "llama_guard_3_1b",
    "meta_llama_guard_2_8b",
    "qwen3_guard_gen_0_6b",
    "qwen3_guard_gen_4b",
    "qwen3_guard_gen_8b",
]

# Files we skip uploading (training-only cruft that may leak absolute paths or is unnecessary).
IGNORE_PATTERNS = [
    "training_args.bin",
    "training_loss.png",
]


def detect_user() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token) if token else HfApi()
        who = api.whoami()
        return who.get("name")
    except Exception:
        pass
    try:
        out = subprocess.check_output(["huggingface-cli", "whoami"], text=True).strip()
        if out and out != "Not logged in":
            return out.splitlines()[0].strip()
    except Exception:
        pass
    return None


def weights_ok(p: Path) -> bool:
    if not p.is_dir():
        return False
    safes = list(p.glob("*.safetensors"))
    return bool(safes) or (p / "pytorch_model.bin").is_file()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", help="HF username/org (auto-detected via whoami if omitted).")
    ap.add_argument("--public", action="store_true", help="Create public repos (default: private).")
    ap.add_argument("--only", action="append", default=[], metavar="SLUG",
                    help="Only upload these slugs (repeatable).")
    ap.add_argument("--prefix", default="sft_gr", help="Repo name prefix (default: sft_gr).")
    ap.add_argument("--base", default=str(BASE), help=f"Weights base dir (default: {BASE}).")
    ap.add_argument("--extra", action="append", default=[], metavar="PATH:REPO_NAME",
                    help="Extra folder to upload as <user>/<REPO_NAME> (repeatable).")
    ap.add_argument("--no-default", action="store_true",
                    help="Skip the default 5 sft_gr slugs (only upload --extra entries).")
    ap.add_argument("--dry-run", action="store_true", help="Don't create repos or upload.")
    args = ap.parse_args()

    extras: list[tuple[Path, str]] = []
    for spec in args.extra:
        if ":" not in spec:
            print(f"ERROR: --extra expects PATH:REPO_NAME, got {spec!r}", file=sys.stderr)
            return 2
        path_str, name = spec.rsplit(":", 1)
        extras.append((Path(path_str).expanduser().resolve(), name.strip()))

    user = args.user or detect_user()
    if not user:
        print("ERROR: cannot detect HF user. Run `huggingface-cli login`, "
              "export HF_TOKEN=..., or pass --user.", file=sys.stderr)
        return 2

    base = Path(args.base)
    slugs = args.only if args.only else (SLUGS if not args.no_default else [])
    private = not args.public

    print(f"HF user   : {user}")
    print(f"Base dir  : {base}")
    print(f"Slugs     : {slugs}")
    print(f"Extras    : {[(str(p), n) for p, n in extras]}")
    print(f"Visibility: {'public' if args.public else 'private'}")
    print(f"Ignore    : {IGNORE_PATTERNS}")
    print()

    missing = [s for s in slugs if not weights_ok(base / s)]
    if missing:
        print(f"WARNING: no safetensors/bin found under: {missing}")
        for s in missing:
            print(f"  - {base / s}")
        slugs = [s for s in slugs if s not in missing]

    missing_extras = [(p, n) for p, n in extras if not weights_ok(p)]
    if missing_extras:
        print(f"WARNING: extras with no safetensors/bin:")
        for p, n in missing_extras:
            print(f"  - {p}  ({n})")
        extras = [(p, n) for p, n in extras if (p, n) not in missing_extras]

    if not slugs and not extras:
        print("Nothing to upload.", file=sys.stderr)
        return 1
    print()

    # Build the final (folder, repo_id) list.
    jobs: list[tuple[Path, str]] = []
    for s in slugs:
        jobs.append((base / s, f"{user}/{args.prefix}-{s}"))
    for p, n in extras:
        jobs.append((p, f"{user}/{n}"))

    if args.dry_run:
        for folder, repo_id in jobs:
            print(f"[dry-run] would upload {folder}  ->  {repo_id}")
        return 0

    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token) if token else HfApi()

    failed: list[tuple[str, str]] = []
    for folder, repo_id in jobs:
        print(f"==> {repo_id}  <-  {folder}")
        try:
            create_repo(repo_id=repo_id, repo_type="model", private=private,
                        exist_ok=True, token=token)
            api.upload_folder(
                folder_path=str(folder),
                repo_id=repo_id,
                repo_type="model",
                ignore_patterns=IGNORE_PATTERNS,
                commit_message=f"Upload {repo_id.split('/')[-1]} checkpoint",
                token=token,
            )
            print(f"    done: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"    FAILED: {e}", file=sys.stderr)
            failed.append((repo_id, str(e)))

    if failed:
        print("\nFailures:", file=sys.stderr)
        for r, e in failed:
            print(f"  - {r}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
