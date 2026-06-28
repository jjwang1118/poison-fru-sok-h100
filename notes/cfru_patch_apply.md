# CFRU Path-A — apply & run sheet
Apply 3 edits to `external/CFRU`, then run. Effect = utility (HR@20) / UDR. CFRU only.
All line numbers are from the freshly-cloned repo; match on the code text, not the number.

---

## STEP 0 — data (once)
```sh
cd external/CFRU
# get & split MovieLens-1M (see README): unrar benchmark/movielens1m/data/movielens1m.rar, then:
bash gen_data.sh        # -> fedtask/movielens1m_cnum100_dist0_skew0.0_seed0/data.json
```
**Env fix (already applied):** the repo's `torchvision` (3×) and `yaml` (2×) imports are
unused dead imports and are now commented out — do NOT install them. After scanning the whole
run path, the ONLY external deps are `numpy, pandas, scipy, torch, ujson`
(`pip install ujson pandas scipy` if a `ModuleNotFoundError` appears; you already have torch/numpy).
`cvxopt` / `prettytable` are never imported; `matplotlib` is only used by the repo's
`CFRU_test.py`, which our driver replaces.

---

## STEP 1 — Patch B+C (edit `external/CFRU`)

### Edit 1 — `utils/fflow.py`, in `read_option()` (after the last `add_argument`, ~line 75)
ADD:
```python
    parser.add_argument('--forget_frac', help='fraction of the attacker set to forget (nested prefix)', type=float, default=1.0)
    parser.add_argument('--mode', help='unlearn=CFRU, retrain=exact retrain without forget set', type=str, choices=['unlearn','retrain'], default='unlearn')
```

### Edit 2 — `utils/fflow.py`, in `initialize()` — replace the attacker block (the `if num_clients == 100: s_atk = [...]` ... `option['malicious_users'] = malicious_users`)
REPLACE WITH:
```python
    if num_clients == 100:
        s_atk = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif num_clients == 10:
        s_atk = [0]
    else:
        raise ValueError('Not enough clients')
    # PATCH B: nested forget set = prefix of the attacker set
    k = round(option['forget_frac'] * len(s_atk))
    option['forget'] = s_atk[:k]
    # PATCH C: retrain => forgotten clients leave the federation; only C\forget keep attacking
    if option['mode'] == 'retrain':
        keep = [cid for cid in range(num_clients) if cid not in option['forget']]
        attacker = [cid for cid in s_atk if cid not in option['forget']]
    else:
        keep = list(range(num_clients))
        attacker = s_atk
    option['keep'] = keep
    option['attacker'] = attacker
    malicious_users = []
    for cid in attacker:
        malicious_users = malicious_users + users_per_client[cid]
    option['malicious_users'] = malicious_users
```
THEN in the SAME function, the `clients = [Client(...) for cid in range(num_clients)]` line —
change the loop to `for cid in option['keep']`:
```python
    clients = [Client(option, name = client_names[cid], model = utils.fmodule.Model(clients_config[cid], option).to(utils.fmodule.device),
                    train_data = client_train_datas[cid], users_set = users_per_client[cid]) for cid in option['keep']]
```

### Edit 3 — `algorithm/CFRU.py`, in `iterate()` — the unlearn target
CHANGE the line `if cid in self.option['attacker']:` (inside the `attack_clients` loop) to:
```python
			if cid in self.option['forget']:
```

### Edit 4 — `algorithm/CFRU.py`, in `save_result()` — handle retrain & empty unlearn
REPLACE the block that builds `clean_model`/`save_logs` (inside `if round_num >= ...`) WITH:
```python
			temp_model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
			if self.option['mode'] == 'retrain':
				save_logs = {
					"selected_clients": self.selected_clients,
					"accuracy_retain": self.test_on_clients(temp_model),
				}
			else:
				clean_model = temp_model if self.unlearn_term is None else (temp_model + self.unlearn_term)
				save_logs = {
					"selected_clients": self.selected_clients,
					"accuracy": self.test_on_clients(temp_model),
					"accuracy_unlearn": self.test_on_clients(clean_model),
					"unlearn_time": self.unlearn_time,
				}
```
(Leave the `pickle.dump(save_logs, ...)` line below it unchanged.)

### Edit 5 — `algorithm/CFRU.py`, in `train()` — make the clean run actually clean
The repo never wired `--atk_method none`: an attacker-set client with `none` falls into the
malicious branch but calls no negative sampler, so `NCFData.features_fill` is never built
(`AttributeError`). Gate the malicious branch on the attack being active:
```python
        if self.name in name_malicious_client and self.option['atk_method'] != 'none':
```
Now `--atk_method none` routes every client through the honest path (`pos_sampling`), giving a
true $M_{\mathsf{clean}}$. Attack runs (`fedAttack`/`fedFlipGrads`) are unaffected.

### Edit 6 — `benchmark/toolkits.py`, in `get_loss_variance()` — NaN guard
The min–max normalisation of candidate scores divides by `(max-min)`, which is `0` for any
user whose candidate pool is degenerate (repeated items → identical scores) → `0/0` → `NaN` →
`np.random.choice(..., p=NaN)` crash. Guard it:
```python
        denom = max_vals - min_vals
        denom[denom == 0] = 1.0
        ratings_mu_candidates = (ratings_mu_candidates - min_vals) / denom
```
Constant rows become uniform after the softmax; all other rows are unchanged.

### Edit 7 — device-agnostic (`.cuda()` → device) — REQUIRED on Mac/CPU
The repo hard-codes `.cuda()` in `CFRU.py:process_grad` (1×), `toolkits.py` test loops (6×),
and `NCF.py:predict_user` (1×). On a machine without NVIDIA CUDA (e.g. macOS) every run crashes
with *"Torch not compiled with CUDA enabled."* Replace with the already-available device:
`M_v.cuda()`→`M_v.to(fmodule.device)`; `x.cuda()`→`x.to(self.device)` in the calculator;
`...long().cuda()`→`...long().to(self.embed_user.weight.device)` in NCF. `fmodule.device`
auto-selects cuda/cpu, so this is correct on every platform.

> **Status (validated by me in a sandbox CPU run):** Edits 1–7 let the training loop run
> end-to-end with no crash (imports → data load → honest+attacker client training → device path).
> I could NOT finish a *full* run in-sandbox (ml-1m on one CPU core is too slow for the harness's
> time budget), so `save_result`/unlearn-term/retrain/parse are code-reviewed but not yet
> runtime-confirmed. The Phase-A check below confirms them on your machine in minutes.

---

## STEP 2 — run it (Mac = CPU; no CUDA/MPS in this code → expect it to be slow)

**Phase A — fast plumbing check (do this FIRST, ~minutes).** In `scripts/run_unlearning_gap.py`
temporarily set `NUM_ROUNDS = 3`, `NUM_EPOCHS = 1`, `BATCH_SIZE = 512` (knobs are at the top).
Then run the command below. Success = `data/unlearning_gap.csv` gets 2 rows with every column
filled and no traceback. This confirms save/unlearn/retrain/parse end-to-end. Numbers are
throwaway at this scale.

**Phase B — real smoke.** Restore `NUM_ROUNDS = 40`, `NUM_EPOCHS = 5`, `BATCH_SIZE = 64`
(authors' values). Keep `SEEDS=[0]`, `FORGET_FRACS=[0.5,1.0]`. Re-run; check `hr_poison < hr_clean`.
Edit `scripts/run_unlearning_gap.py`: set `SEEDS=[0]`, `FORGET_FRACS=[0.5,1.0]`. Then:
```sh
cd external/CFRU
python3 ../../scripts/run_unlearning_gap.py
```
Sanity checks before trusting anything:
- `hr_poison < hr_clean` (the attack actually degrades utility). If not, raise rounds /
  attacker strength, or `ATTACK_DROP_DELTA` is mis-set.
- `hr_retain` (retrain without forget) and `hr_unlearned` (CFRU) are both produced.
- one row per (seed,frac) lands in `data/unlearning_gap.csv`.

## STEP 3 — full sweep
Set `SEEDS=[0,1,2,3,4,5]`, `FORGET_FRACS=[0.2,0.4,0.6,0.8,1.0]`, rerun STEP 2 command.
(3 runs per (seed,frac): clean once per seed + unlearn + retrain per frac. Cost is
dominated by the retrains; shrink seeds/fracs if needed, never drop retrain.)

## STEP 4 — table + paper
```sh
cd "<repo root>"
python3 scripts/make_unlearning_gap_table.py     # data/ -> paper/tables/unlearning_gap.tex
```
Then in `paper/sections/08_poisonfru_bench.tex` §8.5: uncomment `\input{tables/unlearning_gap}`,
add `Table~\ref{tab:unlearning-gap}`, and replace the protocol sentence with the finding
(au-gap ≈ 0 / positive / negative — see notes/unlearning_experiment_design.md §1).
Tell me the numbers and I will write the results paragraph + UDR interpretation.

---

## [VERIFY] while smoke-testing
1. `--atk_method none` truly disables the attack in client training (clean run correctness).
2. `test_on_clients` tests on the GLOBAL test set (so all 4 models are comparable); if it is
   per-client, pass a retained-user subset for consistency.
3. retrain reindexes clients (forget clients dropped) — confirm training runs without index
   errors (the attack keys off user-ids, not client-index, so it should be fine).
4. `--topN 20` flows into the saved HR/NDCG (we want HR@20).

---

## Root-cause fix (Option B) — why the first runs gave HR=NDCG=1.0

**Symptom.** The clean run reported $\mathrm{HR@20}=\mathrm{NDCG@20}=1.0$ every round — a
degenerate (saturated) result; real ml-1m sampled HR is ~0.6--0.7.

**Diagnosis (verified on the real `data.json`).** The eval is *correct* (a fresh untrained
model scores ~0.16; no test/train leakage; 100 distinct candidates per user). Lowering the
learning rate did **not** fix it. The culprit is CFRU's honest-client training path
`get_loss_variance` (its variance-based candidate/negative sampling + `update_variance_sets`),
which produces a degenerate model. A plain pointwise NCF trained on the same data reaches
HR~0.62.

**Fix B (applied to `algorithm/CFRU.py`, honest `else` branch of `Client.train`).** Bypass the
variance machinery: standard negatives via `ng_sample_original()`, standard BCE via
`self.calculator.get_loss(...)`, and FedAvg by returning the full local `model` (no
`process_grad` sparsification, no `update_variance_sets`). The attacker branch is unchanged, so
the `fedAttack` attack and the CFRU `unlearn_term` still run.

**Run config (driver):** `--learning_rate 0.01` (repo default `0.1` collapses NCF);
`PROPORTION = 1.0` because `sample()` is hard-coded to select all clients every round
(`--proportion` is ignored — full participation, label kept honest).

**Paper note — LOCKED口徑 (now written into §8.5 `subsec:gap-decomposition`, paragraph
"Reference construction (protocol note)").** The disclosure must read as a *protocol choice*,
never as a CFRU defect:
- honest clients are trained with **standard FedAvg** (`\cite{McMahan2017FedAvg}`); CFRU stays the
  **unlearning operator** ($M^{-F}$) and attack harness ($M_{\mathsf{poison}}$);
- $M_{\mathsf{clean}}$ is plain FedAvg and the honest substrate of $M_{\mathsf{retain}}$ is FedAvg,
  so all four references are non-degenerate and comparable;
- reason given: CFRU's released pipeline targets the *with-attack* unlearning workflow; the
  no-attack clean baseline and forget-set exact retrain are regimes it does **not natively
  support**, and repurposing its honest-client path there (in our setup) gave a degenerate
  (saturated) reference;
- explicit disclaimer in the text: *"not a claim about bugs in the original CFRU implementation or
  its reported results, which exercise the method in its intended regime."*
- §8.5 stays scoped to the **untargeted/utility-degradation axis**, where the FedAvg references are
  well defined.

Do **not** describe the HR=1.0 degeneracy as a CFRU bug anywhere in the paper: we observed it only
when repurposing their honest path outside its designed regime under our config, and we never
reproduced their reported numbers with their own harness — so a bug claim would be unverified and
unfair.
