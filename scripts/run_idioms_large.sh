#!/usr/bin/env bash
# Idioms (full run) on the two large models the pipeline uses: Gemma 4 31B IT (generator) and
# Qwen 3.8 27B FP8 (reviewer), unmodified. Waits until scripts/run_rn2_sources.sh has exited and the
# GPU has no other processes, then runs each model as its own evaluation so one failure keeps the other.
#   bash scripts/run_idioms_large.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/.local/share/s2s-env.sh  # RESULTS_REPO for the upload
export HF_HUB_CACHE=/scratch/huggingface/hub

while pgrep -fx "bash scripts/run_rn2_sources.sh" >/dev/null; do sleep 300; done
echo "--- rn2 source ablation no longer running: $(date '+%F %H:%M')"
until [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; do
    echo "GPU busy, waiting"; sleep 300
done
if [[ -s /etc/stellar/booking.json ]] && ! python3 -c "
import json, os; b = json.load(open('/etc/stellar/booking.json'))
raise SystemExit(0 if not b or b.get('account') == os.environ['USER'] or b.get('shared') else 1)"; then
    echo "Machine booked by someone else; stopping."; exit 3
fi
for name in gemma4-31b-it qwen3.8-27b-fp8; do
    echo "--- idioms $name: $(date '+%F %H:%M')"
    (cd evaluation && uv run python -m evaluation run --config "idioms-${name}_run.yaml") \
        || echo "!!! idioms $name failed (exit $?)"
done
echo "=== idioms large models finished $(date '+%F %H:%M')"
