# Run the CFRU / UDR experiment on an H100

This bundle is the **patched** CFRU experiment (Option-B honest-path FedAvg,
device-agnostic, nested forget-set driver) plus the MovieLens-1M split, ready to
run on a CUDA GPU. The driver auto-uses the GPU: CFRU sets the device from
`--server_gpu_id 0` whenever `torch.cuda.is_available()`, so no `.cuda()` edits are
needed.

## 1. Environment

Python 3.9+ and a CUDA build of PyTorch:

```
pip install torch numpy pandas scipy ujson      # torch = the CUDA wheel for your driver
nvidia-smi                                       # confirm the H100 is visible
```

## 2. (Recommended) 2-minute smoke to measure GPU speed first

Each round trains ~100 clients sequentially x 5 epochs; the GPU speedup over the
Mac CPU is real but **not** 30x (tiny NCF batches, Python-loop bound). Time a few
rounds before committing to the full sweep:

- edit `scripts/run_unlearning_gap.py`: set `SEEDS = [0]`, `FORGET_FRACS = [1.0]`
- run it (step 3), watch how long one round takes, then restore the full grid
  (`SEEDS = [0,1,2,3,4,5]`, `FORGET_FRACS = [0.2,0.4,0.6,0.8,1.0]`).
- Throughput tip: raising `BATCH_SIZE` (64 -> 256/512) improves GPU utilization a
  lot for NCF; the clean baseline stays healthy.

## 3. Run

```
cd external/CFRU
python3 ../../scripts/run_unlearning_gap.py
```

- **Must** be launched from `external/CFRU` (the benchmark reads
  `./benchmark/movielens1m/data` by a relative path).
- Output: `data/unlearning_gap.csv`, **checkpointed after every (seed,frac)** -
  safe to Ctrl-C or requeue; rerunning **resumes** (skips rows already in the CSV
  and reuses any clean run found on disk).
- Full grid = 6 seeds x 5 fracs = 66 runs (6 clean + 30 unlearn + 30 retrain).
- The split `data.json` is already included; no `gen_data.sh` step needed.

## 4. Sanity checks (before trusting the numbers)

- clean `HR@20` ~ 0.7-0.8 (a healthy ml-1m NCF; the Mac run gave 0.7617).
- `HR_poison < HR_clean` (the untargeted `fedAttack` actually degrades utility).
  If not, the attack did not bite - raise rounds / attacker strength, do **not**
  adjust the metric.

## 5. Send back the result

Just send me `data/unlearning_gap.csv`. I will generate the LaTeX table
(`tables/unlearning_gap.tex`) and write the SS 8.5 results paragraph
(id-gap / au-gap / UDR + the three-way interpretation).

## What changed vs upstream CFRU (so results are reproducible)

- Honest clients train with **standard FedAvg** (Option B); CFRU's selective-update
  machinery stays on the attacker / unlearning path only. This is a protocol choice
  for non-degenerate `M_clean` / `M_retain` references, not a fix to a CFRU bug.
  See `notes/cfru_patch_apply.md`.
- `--learning_rate 0.01` (the repo default `0.1` collapses NCF), `--proportion 1.0`
  (full participation), `R40`.
- Device-agnostic (`.cuda()` -> `.to(device)`); on the H100 it runs on `cuda:0`.
