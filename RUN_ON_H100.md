# Run the CFRU experiment on an H100 — TARGETED / ER@K axis (PSMU)

This bundle is the patched CFRU experiment set up for a **targeted target-item
promotion** attack, with **ER@K** (exposure rate of the target item) as the primary
metric. HR/NDCG (utility) are also recorded so utility-vs-exposure can be contrasted.
The driver auto-uses the GPU (`--server_gpu_id 0` + `torch.cuda`).

**Attack = `psmu` (Route A).** PSMU (Yuan 2023) synthetic-user *model*-poisoning: each
malicious client learns synthetic user embeddings that "like" a random item cluster,
builds a competition set (the target's nearest items), and gradient-descends the
target-item embedding on an exposure objective so the target outranks the competition.
The crafted embedding is written into the **uploaded model** (`cp["model"]`) — the
quantity CFRU actually FedAvg-aggregates — so it reaches the global model and ER.

> Why the previous `targetPromo` gave `ER_poison = 0`: CFRU aggregates the **full**
> client models (`self.aggregate(models)`), and the importance-sparsified
> `important_weights` feed **only** the unlearn term (`Server.process_grad ->
> grads_all_round -> compute_unlearn_term`). So data-level positive injection was
> diluted by full-model FedAvg and never saturated ER. PSMU instead poisons the
> uploaded model directly. `force_keep=[target]` is retained so the same target row
> also survives into the stored update, keeping the unlearn term consistent.

## 0. (optional, no GPU) sanity-check the craft first
```
python3 scripts/test_psmu_craft.py        # from repo root
```
Expect ER@20 to rise from ~0 to ~1.0 as the craft (trained on SYNTHETIC users)
generalizes to REAL users. This validates the mechanism before spending GPU time.

## 1. Environment
```
pip install torch numpy pandas scipy ujson      # torch = the CUDA wheel for your driver
nvidia-smi
```

## 2. SMOKE FIRST — decisive go/no-go (a few runs)
All grid/knob values are env-overridable. Run the full-forget smoke with debug on:
```
cd external/CFRU
ATK_DEBUG=1 SEEDS=0 FORGET_FRACS=1.0 NUM_ROUNDS=40 PSMU_SCALE=5 \
  python3 ../../scripts/run_unlearning_gap.py
```
This does clean + poison(unlearn) + retrain at full forget. Watch the startup line
`target_item = <id>` and the per-round `[ATK] ... er(poison)=...` lines.

**The whole experiment hinges on this check:**
```
ER clean  ~ 0.0       (cold target is not recommended)
ER poison >> ER clean (target promoted; ideally >= 0.3-0.5, saturating toward 1.0)
ER unlearn (full forget) drops back toward ER clean   (CFRU removes what it added)
HR_*      ~ unchanged (utility flat, exposure moves — the headline contrast)
```
If `ER poison` stays low, **raise `PSMU_SCALE`** (the primary knob) to overcome the
~10% malicious-weight FedAvg dilution: try `PSMU_SCALE=10`, then `20`. Secondary knobs
(env / CLI): `PSMU_S` (synthetic users/round, default 8), `--psmu_comp_k`,
`--psmu_item_steps`, `--psmu_lr`. Do **not** lower the `ATTACK_ER_MIN=0.30` gate.

## 3. Run
```
cd external/CFRU
python3 ../../scripts/run_unlearning_gap.py
```
- Output: `data/unlearning_gap.csv`, **checkpointed after every (seed,frac)**; rerun **resumes**.
- Columns: `er_clean / er_poison / er_retain / er_unlearned` (+ `hr_*`, `ndcg_*`, `target_item`, `attack_saturated`).
- Full grid (after the smoke passes, with the tuned scale):
  ```
  SEEDS=0,1,2,3,4,5 FORGET_FRACS=0.2,0.4,0.6,0.8,1.0 NUM_ROUNDS=40 PSMU_SCALE=<tuned> \
    python3 ../../scripts/run_unlearning_gap.py
  ```
- `BATCH_SIZE`: raise to 256/512 for GPU throughput.
- Must be launched from `external/CFRU` (benchmark reads `./benchmark/movielens1m/data` by relative path).
- To compare against the old (diluted) attack: prepend `ATTACK=targetPromo`.

## 4. What the result should show (the paper's point)
At **partial** forget (the forget set F does not cover the whole carrier): `M^{-F}` is close
to `M_retain` on HR and ER (retraining-reference fidelity holds) yet ER stays high
(attack-effect recovery fails). At **full** forget, ER should drop toward clean.

## 5. Send back
Send `data/unlearning_gap.csv` (and the smoke `.out`, including the `[ATK]` lines and the
tuned `PSMU_SCALE`). I build the LaTeX table and write the §8.5 results.

## What changed vs upstream CFRU (reproducibility)
- **New attack `psmu`** (`algorithm/CFRU.py`: `_psmu_attack` + `_psmu_craft_target`,
  registered in `utils/fflow.py`): synthetic-user model-poisoning written into the
  uploaded/aggregated model. Hyper-parameters `--psmu_scale/_s/_user_steps/_item_steps/
  _lr/_comp_k/_std`. The prior `targetPromo` (data injection) is left intact and selectable.
- **New metric ER@K** (`test_target_exposure`): fraction of test users with the target
  item in their top-K. Saved as `er` / `er_unlearn` / `er_retain` in each run's pkl.
- Target item auto-picked as the least-popular item (`--target_item -1`) so clean
  exposure ~ 0; printed at startup and saved in the CSV.
- Honest clients train with standard FedAvg; CFRU's machinery stays on the attacker/unlearn path.
- Device-agnostic; runs on `cuda:0`.

## Honesty / disclosure (for the paper)
- This is an **author-implemented** PSMU-on-CFRU attack, not a CFRU-native attack.
  It is faithful to PSMU's mechanism (synthetic users + competition set + exposure
  gradient on the shared item embedding) but adapts it to CFRU's NeuMF.
- Deviations to disclose: the exposure objective uses a BPR form
  (`-logsigmoid(pos-neg)`) for stable gradients instead of the reference
  `sigmoid(sum(neg-pos))`; only the **target row** is poisoned (linear-layer
  poisoning from the reference impl is omitted); `psmu_scale` amplifies the uploaded
  shift and is tuned to reach the ER gate. All three are marked `[CHECK]` in the code.
