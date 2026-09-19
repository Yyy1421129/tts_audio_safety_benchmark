#!/usr/bin/env bash
set -euo pipefail
cd /workspace
KEY_FILE=/workspace/secrets/kimi_api_key.txt
OUT_ROOT=results/stage4/kimi_judge
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/batch_run.log"
exec > >(tee -a "$LOG") 2>&1

run_one() {
  local dir="$1" label="$2"
  echo "======== START $(date -Is) $label ($dir) ========"
  python3 scripts/evaluate_adv_attack_with_kimi.py \
    --api-key-file "$KEY_FILE" \
    --attack-dir "$dir" \
    --manifest manifests/matcha_seed_prompts_v0_3.jsonl \
    --output "$OUT_ROOT/${label}.jsonl" \
    --summary "$OUT_ROOT/${label}_summary.csv" \
    --label "$label" \
    --request-sleep 22 \
    --max-tokens 512 \
    --max-response-chars 1500
  echo "======== DONE $(date -Is) $label ========"
  # cost so far
  python3 - <<PY
import json
from pathlib import Path
root=Path('$OUT_ROOT')
pin=pc=0; n=0
for p in sorted(root.glob('*.jsonl')):
    if p.name.endswith('pilot.jsonl'):
        continue
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        u=json.loads(line).get('usage') or {}
        pin+=u.get('prompt_tokens',0); pc+=u.get('completion_tokens',0); n+=1
cost=pin*6.5/1e6 + pc*27/1e6
print(f'[cost-so-far] judged={n} prompt={pin} completion={pc} est_yuan={cost:.3f}')
PY
  sleep 25
}

# Skip failed pilot rows from being counted as main; keep pilot files aside
run_one output/adv_attack_results_a10_v4 no_l2
run_one output/adv_attack_v5_l2 l2
run_one output/adv_attack_v8_fixed energy_range

echo "ALL_DONE $(date -Is)"
