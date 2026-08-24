#!/usr/bin/env bash
# run_pipeline.sh — Full feasibility pipeline (Steps 1-5)
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="configs/config.yaml"
MODE="${1:---mock}"  # --mock (no GPU), --real, --debug

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=== Hierarchical NLI E-first — Pipeline Mode=$MODE ==="
echo "CONFIG=$CONFIG"

# 0. ensure data
python3 -c "from src.data.dataset import load_or_generate; load_or_generate('$CONFIG')"

# 1. Flat baseline
echo "--- Step1 Flat 3-class ---"
python3 -m src.training.train_flat --config "$CONFIG" $MODE

# 2. diagnostic done inside train_flat; also offline
# 3. Hierarchical multi-task
echo "--- Step3 Hierarchical ---"
python3 -m src.training.train_hierarchical --config "$CONFIG" $MODE

# 4-5 offline
echo "--- Steps 2,4A,4B,5 offline ---"
python3 scripts/evaluate_offline.py --config "$CONFIG"

echo "=== PIPELINE DONE ==="
echo "Outputs:"
echo "  Flat preds:  outputs/predictions/flat/"
echo "  Hier preds:  outputs/predictions/hier/"
echo "  Metrics:     outputs/metrics/"
echo "  Diagnostics: outputs/diagnostics/"
echo "  Comparison:  outputs/comparison/comparison_report.md"
ls -lh outputs/predictions/flat/ outputs/predictions/hier/ outputs/metrics/ outputs/comparison/ 2>&1 | head -n 60
