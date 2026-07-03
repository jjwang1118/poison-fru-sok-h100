# Run the CFRU experiment on an H100 — TARGETED / ER@K axis

This bundle is the patched CFRU experiment set up for a **targeted target-item
promotion** attack, with **ER@K** (exposure rate of the target item) as the primary
metric. HR/NDCG (utility) are also recorded so utility-vs-exposure can be contrasted.
The driver auto-uses the GPU (`--server_gpu_id 0` + `torch.cuda`).

## 1. Environment
```
pip install torch numpy pandas scipy ujson      # torch = the CUDA wheel for your driver
nvidia-smi
```

## 2. SMOKE FIRST — decisive go/no-go (a few runs)
In `scripts/run_unlearning_gap.py` set:
```
SEEDS        = [0]
FORGET_FRACS = [1.0]
```
then run (step 3). This does clean + poison(unlearn) + retrain at full forget. Watch
the startup line `target_item = <id>` and the final CSV row.

**The whole experiment hinges on this check:**
```
ER clean  ~ 0.0       (cold target is not recommended)
ER poison >> ER clean (target promoted; ideally >= 0.3-0.5)
```
If `ER poison` stays ~ `ER clean`, the attack did not bite — tell me before any sweep;
we raise the malicious fraction (`s_atk` in `utils/fflow.py`) or the injection strength
(the `num_ng` copies in `ng_sample_target`). Utility (HR) should be ~unchanged across the
four models — that contrast (utility flat, exposure differs) is the point.

## 3. Run
```
cd external/CFRU
python3 ../../scripts/run_unlearning_gap.py
```
- Output: `data/unlearning_gap.csv`, **checkpointed after every (seed,frac)**; rerun **resumes**.
- Columns now include `er_clean / er_poison / er_retain / er_unlearned` (+ `hr_*`, `ndcg_*`, `target_item`).
- Full grid (after the smoke passes): `SEEDS=[0,1,2,3,4,5]`, `FORGET_FRACS=[0.2,0.4,0.6,0.8,1.0]`.
- `BATCH_SIZE`: raise to 256/512 for GPU throughput.
- Must be launched from `external/CFRU` (benchmark reads `./benchmark/movielens1m/data` by relative path).

## 4. What the result should show (the paper's point)
At **partial** forget (the forget set F does not cover the whole carrier): `M^{-F}` is close
to `M_retain` on HR and ER (retraining-reference fidelity holds) yet ER stays high
(attack-effect recovery fails). At **full** forget, ER should drop toward clean.

## 5. Send back
Just send me `data/unlearning_gap.csv` (and the smoke `.out`). I build the LaTeX table and
write the SS 8.5 results.

## What changed vs upstream CFRU (reproducibility)
- New attack `targetPromo`: malicious clients inject `(their users, target_item)` as positives
  (`ng_sample_target`), promoting the shared target-item embedding. The target item is
  auto-picked as the least-popular item (`--target_item -1`) so clean exposure ~ 0; it is
  printed at startup and saved in the CSV.
- New metric ER@K (`test_target_exposure`): fraction of test users with the target item in
  their top-K. Saved as `er` / `er_unlearn` / `er_retain` in each run's pkl.
- Honest clients train with standard FedAvg; CFRU's machinery stays on the attacker/unlearn path.
- Device-agnostic; runs on `cuda:0`.

This targeted probe is a **transparent minimal promotion attack** (PipAttack / FedRecAttack
family) used to probe the carrier-vs-forget-set geometry — not a claim about the strongest
possible attack.
```
