#!/usr/bin/env bash
# Compute per-line confidence on train_all generated_predictions.jsonl for:
#   results/rg_final/<slug>/train_all                ← saves/rg_final/<slug>
#   results/rg_final/subset_sft/<slug>/train_all     ← saves/rg_final/subset_sft_expert/<slug>
# Parallelism: one worker per GPU in GPUS (default 0 1 2 3).
# Output (alongside input): generated_predictions_conf.jsonl
# Logs:                     results/rg_final/_conf_logs/<tag>_<UTCstamp>.log
#
# Usage:
#   bash examples/test_small_guard/run_guard_confidence_train_all.sh
#   GPUS="0 1 2 3"      bash ...            # override GPU list
#   BATCH_SIZE=128      bash ...
#   RESUME=1            bash ...            # append after existing output
#   RUN_ONLY="agent social"  bash ...       # whitespace-separated slug list
#   RUN_GROUPS="rg_final"   bash ...        # only rg_final/<slug>
#   RUN_GROUPS="subset_sft" bash ...        # only subset_sft/<slug>
#   DRY_RUN=1           bash ...            # print planned jobs and exit
#
# One failure does not stop the rest; non-zero exit if any failed.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate dl 2>/dev/null || true
fi

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Don't inherit CUDA_VISIBLE_DEVICES — we set it per worker.
unset CUDA_VISIBLE_DEVICES

SLUGS_DEFAULT=(agent cyber harm non_violent social)
if [[ -n "${RUN_ONLY:-}" ]]; then
  # shellcheck disable=SC2206
  SLUGS=(${RUN_ONLY})
else
  SLUGS=("${SLUGS_DEFAULT[@]}")
fi

# NOTE: $GROUPS is a bash builtin array (user's group IDs); use RUN_GROUPS instead.
RUN_GROUPS="${RUN_GROUPS:-rg_final subset_sft}"
BATCH_SIZE="${BATCH_SIZE:-8}"   # lm_head over [B, L, vocab=152k] in bf16 blows up VRAM fast; raise only if you checked.
# shellcheck disable=SC2206
GPUS=(${GPUS:-0 1 2 3})
RESUME_FLAG=""
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG="--resume"
fi

SCRIPT="examples/test_small_guard/compute_guard_confidence.py"
LOG_DIR="results/rg_final/_conf_logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

TASKS=()
FAILED=()

queue_task() {
  local tag="$1" model="$2" input="$3"
  if [[ ! -d "${model}" ]]; then
    echo "SKIP ${tag}: missing model dir ${model}" >&2
    FAILED+=("${tag} (missing model)")
    return
  fi
  if [[ ! -f "${input}" ]]; then
    echo "SKIP ${tag}: missing input ${input}" >&2
    FAILED+=("${tag} (missing input)")
    return
  fi
  # Use ASCII unit separator (0x1f): '|' can appear inside paths or confuse read on some setups.
  TASKS+=("${tag}"$'\x1f'"${model}"$'\x1f'"${input}")
}

for g in ${RUN_GROUPS}; do
  case "${g}" in
    rg_final)
      for slug in "${SLUGS[@]}"; do
        queue_task "rg_final/${slug}" \
          "/nas02/jacky/Debug_LM/saves/rg_final/${slug}" \
          "/nas02/jacky/Debug_LM/results/rg_final/${slug}/train_all/generated_predictions.jsonl"
      done
      ;;
    subset_sft)
      for slug in "${SLUGS[@]}"; do
        queue_task "subset_sft/${slug}" \
          "/nas02/jacky/Debug_LM/saves/rg_final/subset_sft_expert/${slug}" \
          "/nas02/jacky/Debug_LM/results/rg_final/subset_sft/${slug}/train_all/generated_predictions.jsonl"
      done
      ;;
    *)
      echo "Unknown group: ${g} (expected: rg_final | subset_sft)" >&2
      FAILED+=("group:${g}")
      ;;
  esac
done

if ((${#TASKS[@]} == 0)); then
  echo "No tasks to run." >&2
  ((${#FAILED[@]} > 0)) && { echo "Failed/skipped: ${FAILED[*]}" >&2; exit 1; }
  exit 0
fi

echo "Planned ${#TASKS[@]} task(s) on GPUs: ${GPUS[*]}  (batch=${BATCH_SIZE}  resume=${RESUME:-0})"
for t in "${TASKS[@]}"; do
  IFS=$'\x1f' read -r tg mo ip <<<"$t"
  printf '  - %-28s model=%s\n                              input=%s\n' "${tg}" "${mo}" "${ip}"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "(DRY_RUN=1; not launching.)"
  exit 0
fi

# --- GPU worker pool via FIFO -------------------------------------------------
FIFO="$(mktemp -u /tmp/conf_fifo.XXXXXX)"
mkfifo "${FIFO}"
exec 9<>"${FIFO}"
rm -f "${FIFO}"

STATUS_DIR="$(mktemp -d /tmp/conf_status.XXXXXX)"
trap 'exec 9>&- 2>/dev/null || true; rm -rf "${STATUS_DIR}" 2>/dev/null || true' EXIT

FIFO_LOCK="${STATUS_DIR}/fifo.lock"
: > "${FIFO_LOCK}"

worker() {
  local gpu="$1"
  local line
  while :; do
    line=""
    # Serialize FIFO reads: bash `read` reads byte-by-byte to stop at '\n',
    # so multiple parallel readers on the same fd will interleave characters.
    {
      flock -x 8
      IFS= read -r line <&9
    } 8>"${FIFO_LOCK}"
    local rc=$?
    (( rc != 0 )) && break                  # read returned non-zero (unexpected) -> stop
    [[ "${line}" == "__EOF__" ]] && break   # sentinel -> this worker is done
    [[ -z "${line}" ]] && continue

    IFS=$'\x1f' read -r tag model input <<<"${line}"
    if [[ -z "${model}" || -z "${input}" ]] || [[ ! -d "${model}" ]] || [[ ! -f "${input}" ]]; then
      echo "$(date -u) [gpu=${gpu}] SKIP bad task line (parse/path): tag='${tag}' model='${model}' input='${input}'" >&2
      echo "${tag:-bad_line}" >> "${STATUS_DIR}/failed"
      continue
    fi
    local safe_tag="${tag//\//__}"
    local log="${LOG_DIR}/${safe_tag}_${STAMP}.log"
    echo "$(date -u) [gpu=${gpu}] START ${tag}  log=${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT}" \
      --model "${model}" \
      --input-jsonl "${input}" \
      --batch-size "${BATCH_SIZE}" \
      ${RESUME_FLAG} >"${log}" 2>&1
    rc=$?
    if (( rc != 0 )); then
      echo "$(date -u) [gpu=${gpu}] FAIL  ${tag} rc=${rc}" >&2
      echo "${tag}" >> "${STATUS_DIR}/failed"
    else
      echo "$(date -u) [gpu=${gpu}] DONE  ${tag}"
    fi
  done
}

WORKER_PIDS=()
for gpu in "${GPUS[@]}"; do
  worker "${gpu}" &
  WORKER_PIDS+=("$!")
done

for task in "${TASKS[@]}"; do
  printf '%s\n' "${task}" >&9
done
# One sentinel per worker so each one exits cleanly (can't rely on EOF:
# fd 9 is opened read-write in this shell, so children see no EOF).
for _ in "${GPUS[@]}"; do
  printf '__EOF__\n' >&9
done
exec 9>&-

for pid in "${WORKER_PIDS[@]}"; do
  wait "${pid}" || true
done

if [[ -f "${STATUS_DIR}/failed" ]]; then
  while IFS= read -r t; do FAILED+=("${t}"); done < "${STATUS_DIR}/failed"
fi

echo "$(date -u) Done."
if ((${#FAILED[@]} > 0)); then
  echo "$(date -u) Failed: ${FAILED[*]}" >&2
  exit 1
fi
