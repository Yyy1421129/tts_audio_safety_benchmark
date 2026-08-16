#!/usr/bin/env bash
set -euo pipefail

# Run Step-Audio2 phase-2 shards in parallel inside a single multi-GPU job.
# Each shard is bound to one GPU by CUDA_VISIBLE_DEVICES.

NUM_SHARDS="${NUM_SHARDS:-4}"
MANIFEST="${MANIFEST:-manifests/phase2_audio_perturbations.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/stepaudio2_phase2_perturbations/shards}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${OUTPUT_DIR}"

PIDS=()
TOTAL_PADDED="$(printf "%02d" "${NUM_SHARDS}")"

echo "NUM_SHARDS=${NUM_SHARDS}"
echo "MANIFEST=${MANIFEST}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

for ((SHARD_INDEX=0; SHARD_INDEX<NUM_SHARDS; SHARD_INDEX++)); do
  SHARD_PADDED="$(printf "%02d" "${SHARD_INDEX}")"
  OUTPUT="${OUTPUT_DIR}/shard_${SHARD_PADDED}_of_${TOTAL_PADDED}.jsonl"
  LOG="${OUTPUT_DIR}/shard_${SHARD_PADDED}_of_${TOTAL_PADDED}.log"

  OVERWRITE_FLAG=""
  if [[ "${OVERWRITE}" == "1" ]]; then
    OVERWRITE_FLAG="--overwrite"
  fi

  echo "Launching shard ${SHARD_INDEX}/${NUM_SHARDS} on GPU ${SHARD_INDEX}: ${OUTPUT}"
  (
    export CUDA_VISIBLE_DEVICES="${SHARD_INDEX}"
    python3 scripts/run_stepaudio2_on_matcha_prompts.py \
      --manifest "${MANIFEST}" \
      --output "${OUTPUT}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${SHARD_INDEX}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      ${OVERWRITE_FLAG}
  ) > "${LOG}" 2>&1 &
  PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    STATUS=1
  fi
done

if [[ "${STATUS}" -ne 0 ]]; then
  echo "At least one shard failed. Check logs under ${OUTPUT_DIR}." >&2
  exit "${STATUS}"
fi

echo "All shards completed successfully."
