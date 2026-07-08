#!/bin/bash
#SBATCH --account=mst115223
#SBATCH --job-name=poison_smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=smoke-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=114423002@cc.ncu.edu.tw

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OUTCSV="$HOME/poison-fru-sok-h100/data/smoke_targetPromo.csv"

cd ~/poison-fru-sok-h100/external/CFRU
source ~/miniconda3/etc/profile.d/conda.sh
conda activate poison-fru-sok

ATK_DEBUG=1 SEEDS=0 FORGET_FRACS=1.0 NUM_ROUNDS=40 PSMU_SCALE=5 \
  python3 ../../scripts/run_unlearning_gap.py

echo "=== smoke result (look for ER poison >> ER clean) ==="
cat "$OUTCSV"
