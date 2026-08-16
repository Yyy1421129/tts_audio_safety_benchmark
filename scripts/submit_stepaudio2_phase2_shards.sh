#!/usr/bin/env bash
set -euo pipefail

# Submit one multi-GPU Step-Audio2 inference job for the phase-2 perturbation manifest.
# The submitted job launches one shard process per GPU inside the container.
#
# Usage examples:
#   bash scripts/submit_stepaudio2_phase2_shards.sh --num-shards 8 --dry-run
#   bash scripts/submit_stepaudio2_phase2_shards.sh --num-shards 8
#   bash scripts/submit_stepaudio2_phase2_shards.sh --num-shards 8 --overwrite

NUM_SHARDS=4
PARTITION="pdgpu-a10"
IMAGE="docker.v2.aispeech.com/sjtu/sjtu_yukai-yiyang-stepaudio2:v1"
CPU=16
MEM="96G"
GPU=""
NODES=1
PROJECT_ROOT="/hpc_stor03/sjtu_home/yi.yang/tts_audio_safety_benchmark_plan"
MANIFEST="manifests/phase2_audio_perturbations.jsonl"
OUTPUT_DIR="results/stepaudio2_phase2_perturbations/shards"
MAX_NEW_TOKENS=512
DRY_RUN=0
OVERWRITE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-shards) NUM_SHARDS="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --cpu) CPU="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${GPU}" ]]; then
  GPU="${NUM_SHARDS}"
fi

mkdir -p "${PROJECT_ROOT}/${OUTPUT_DIR}"

JOB_CMD="cd ${PROJECT_ROOT} && NUM_SHARDS=${NUM_SHARDS} MANIFEST=${MANIFEST} OUTPUT_DIR=${OUTPUT_DIR} MAX_NEW_TOKENS=${MAX_NEW_TOKENS} OVERWRITE=${OVERWRITE} bash scripts/run_stepaudio2_phase2_shards_local.sh"
SUBMIT_CMD=(vc submit -p "${PARTITION}" -i "${IMAGE}" -c "${CPU}" -m "${MEM}" -g "${GPU}" -n "${NODES}" --cmd "${JOB_CMD}")

printf "%q " "${SUBMIT_CMD[@]}"
printf "\n"
if [[ "${DRY_RUN}" -eq 0 ]]; then
  "${SUBMIT_CMD[@]}"
fi
