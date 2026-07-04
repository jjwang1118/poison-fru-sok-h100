#!/bin/bash
#SBATCH --account=mst115223
#SBATCH --job-name=poison_smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=smoke-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=114423002@cc.ncu.edu.tw

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# SMOKE = small, fast go/no-go for the TARGETED attack.
# It runs ONE seed at FULL forget = 3 runs total (clean + poison + retrain).
# Goal: confirm  ER(poison) >> ER(clean) ~ 0  BEFORE committing to the full sweep.
# The grid below is passed to scripts/run_unlearning_gap.py via env vars,
# so you never have to edit the Python file.
# ---------------------------------------------------------------------------
export SEEDS="0"               # one seed
export FORGET_FRACS="1.0"      # full forget only (1 fraction)
export NUM_ROUNDS="40"         # quick peek; set "40" to match the full run exactly
export BATCH_SIZE="512"        # big batch -> better GPU throughput
export ATTACK="targetPromo"    # targeted target-item promotion (ER@K axis)
export OUTCSV="$HOME/poison-fru-sok-h100/data/smoke_targetPromo.csv"   # separate from the full-run CSV

cd ~/poison-fru-sok-h100/external/CFRU
source ~/miniconda3/etc/profile.d/conda.sh
conda activate poison-fru-sok
python3 ../../scripts/run_unlearning_gap.py

echo "=== smoke result (look for ER poison >> ER clean) ==="
cat "$OUTCSV"

# ---------------------------------------------------------------------------
# IF the smoke passes (ER poison clearly above ER clean), run the FULL sweep
# with the paper config by simply NOT setting the env vars, e.g.:
#
#   cd ~/poison-fru-sok-h100/external/CFRU
#   ATTACK=targetPromo python3 ../../scripts/run_unlearning_gap.py
#
# (defaults: SEEDS=0..5, FORGET_FRACS=0.2..1.0, NUM_ROUNDS=40, writes
#  data/unlearning_gap.csv, checkpointed + resumable).
# ---------------------------------------------------------------------------
