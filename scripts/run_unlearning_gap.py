#!/usr/bin/env python3
"""Driver for the Path-A (utility-degradation / UDR) algorithm-level gap experiment.

It orchestrates the (patched) CFRU CLI and parses each run's
`fedtasksave/<task>/<exp>/record/history{R}.pkl` to build per-seed rows in
`data/unlearning_gap.csv`.

Per (seed, forget_frac) it launches THREE runs:
  clean   : --atk_method none     --forget_frac 0  --mode unlearn   -> accuracy            = HR/NDCG(M_clean)
  unlearn : --atk_method fedAttack --forget_frac f  --mode unlearn   -> accuracy            = HR/NDCG(M_poison)
                                                                        accuracy_unlearn     = HR/NDCG(M^{-F})
  retrain : --atk_method fedAttack --forget_frac f  --mode retrain   -> accuracy_retain      = HR/NDCG(M_retain)

Requires Patches B and C applied to external/CFRU (see notes/cfru_patch_apply.md).
Run it FROM inside external/CFRU (so relative paths match the repo), e.g.:
    cd external/CFRU && python3 ../../scripts/run_unlearning_gap.py
Smoke test first: set SEEDS=[0], FORGET_FRACS=[0.5,1.0].
"""
from __future__ import annotations

import csv, io, os, pickle, subprocess, sys
from pathlib import Path

# ---- experiment grid (edit me) ---------------------------------------------
ANCHOR        = "CFRU-fedAttack-ML1M-NCF"
TASK          = "movielens1m_cnum100_dist0_skew0_seed0"
MODEL         = "NCF"
NUM_ROUNDS    = 40                       # full training (B root-cause fix: honest path = plain FedAvg)
TOPN          = 20                       # HR@20 / NDCG@20
S1            = 8                         # repo's S1 (appears in the exp folder name)
PROPORTION    = 1.0                      # sample() uses ALL clients (full participation); 1.0 = honest label
ALPHA         = 0.5
LEARNING_RATE = 0.01                     # repo default 0.1 collapses NCF; 0.01 is healthy (root-cause fix)
NUM_EPOCHS    = 5                         # authors' default; set 1 for a fast plumbing check
BATCH_SIZE    = 256                        # authors' default; set 512 for a fast plumbing check
NUM_NG        = 4                         # negatives per positive (authors' default)
SEEDS         = [0]        # full sweep (H100: fast; checkpoint/resume make it safe) [0, 1, 2, 3, 4, 5] 
FORGET_FRACS  = [1.0] # full nested sweep [0.2, 0.4, 0.6, 0.8, 1.0]
GPU_ID        = 0                          # --server_gpu_id; CFRU auto-uses cuda:GPU_ID when torch.cuda is available
PY            = sys.executable

REPO   = Path(__file__).resolve().parents[1] / "external" / "CFRU"
OUTCSV = Path(__file__).resolve().parents[1] / "data" / "unlearning_gap.csv"


class _CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def exp_dir(seed: int, atk: str) -> Path:
    # mirrors algorithm/CFRU.py self.path_save
    name = "CFRU_{}_Mu{}_R{}_P{:.2f}_alpha{}_seed{}_{}".format(
        MODEL, S1, NUM_ROUNDS, PROPORTION, ALPHA, seed, atk)
    return REPO / "fedtasksave" / TASK / name / "record"


def run(seed: int, atk: str, forget_frac: float, mode: str) -> dict:
    cmd = [PY, "main.py", "--task", TASK, "--model", MODEL, "--algorithm", "CFRU",
           "--atk_method", atk, "--mode", mode, "--forget_frac", str(forget_frac),
           "--num_rounds", str(NUM_ROUNDS), "--topN", str(TOPN), "--S1", str(S1),
           "--proportion", str(PROPORTION), "--alpha", str(ALPHA), "--seed", str(seed),
           "--num_epochs", str(NUM_EPOCHS), "--batch_size", str(BATCH_SIZE), "--num_ng", str(NUM_NG),
           "--learning_rate", str(LEARNING_RATE), "--server_gpu_id", str(GPU_ID)]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)
    pkl = exp_dir(seed, atk) / f"history{NUM_ROUNDS}.pkl"
    with open(pkl, "rb") as f:
        return _CPU_Unpickler(f).load()      # dict with accuracy / accuracy_unlearn / accuracy_retain


def load_pkl(seed: int, atk: str):
    """Reuse an existing run's result from disk (used to skip re-running a clean run)."""
    pkl = exp_dir(seed, atk) / f"history{NUM_ROUNDS}.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            return _CPU_Unpickler(f).load()
    return None


FIELDS = ["anchor", "method", "seed", "forget_frac", "attack_saturated",
          "hr_clean", "hr_poison", "hr_retain", "hr_unlearned",
          "ndcg_clean", "ndcg_poison", "ndcg_retain", "ndcg_unlearned"]
ATTACK_DROP_DELTA = 0.02   # 'saturated' = poison drops HR by >= this vs clean (tune from smoke)


def _read_existing():
    """Resume support: return {(seed,frac): row}, {seed: (hr,ndcg)}, header_ok."""
    done, clean = {}, {}
    if OUTCSV.exists() and OUTCSV.stat().st_size > 0:
        with OUTCSV.open(newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            if rd.fieldnames != FIELDS:
                return done, clean, False            # stale/incompatible header -> overwrite
            for r in rd:
                try:
                    s, fr = int(r["seed"]), float(r["forget_frac"])
                except (KeyError, ValueError):
                    continue
                done[(s, fr)] = r
                clean[s] = (float(r["hr_clean"]), float(r["ndcg_clean"]))
    return done, clean, True


def main() -> int:
    # Resume + per-(seed,frac) checkpoint: a crash/requeue never loses finished runs.
    done, clean_cache, header_ok = _read_existing()
    fresh = not (OUTCSV.exists() and OUTCSV.stat().st_size > 0 and header_ok)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    fout = OUTCSV.open("w" if fresh else "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=FIELDS)
    if fresh:
        writer.writeheader(); fout.flush()
        done, clean_cache = {}, {}

    n_new = 0
    for seed in SEEDS:
        # clean once per seed: CSV cache -> on-disk pkl -> run
        if seed in clean_cache:
            hr_c, ndcg_c = clean_cache[seed]
            print(f"  [skip]  seed={seed} clean (from CSV): HR={hr_c:.4f}")
        else:
            rec = load_pkl(seed, "none")
            if rec and "accuracy" in rec:
                hr_c, ndcg_c = rec["accuracy"]
                print(f"  [reuse] seed={seed} clean (disk pkl): HR={hr_c:.4f}")
            else:
                hr_c, ndcg_c = run(seed, "none", 0.0, "unlearn")["accuracy"]
            clean_cache[seed] = (hr_c, ndcg_c)

        for frac in FORGET_FRACS:
            if (seed, frac) in done:
                print(f"  [skip]  seed={seed} frac={frac} (already in CSV)")
                continue
            ru = run(seed, "fedAttack", frac, "unlearn")        # accuracy=poison, accuracy_unlearn=M^{-F}
            rr = run(seed, "fedAttack", frac, "retrain")        # accuracy_retain=M_retain
            hr_p, ndcg_p = ru["accuracy"]
            hr_u, ndcg_u = ru["accuracy_unlearn"]
            hr_r, ndcg_r = rr["accuracy_retain"]
            row = {
                "anchor": ANCHOR, "method": "CFRU", "seed": seed, "forget_frac": frac,
                "attack_saturated": int((hr_c - hr_p) >= ATTACK_DROP_DELTA),
                "hr_clean": hr_c, "hr_poison": hr_p, "hr_retain": hr_r, "hr_unlearned": hr_u,
                "ndcg_clean": ndcg_c, "ndcg_poison": ndcg_p, "ndcg_retain": ndcg_r, "ndcg_unlearned": ndcg_u,
            }
            writer.writerow(row); fout.flush()                  # checkpoint after each (seed,frac)
            done[(seed, frac)] = row; n_new += 1
            print(f"  -> seed={seed} frac={frac}: HR clean={hr_c:.4f} poison={hr_p:.4f} "
                  f"retain={hr_r:.4f} unlearn={hr_u:.4f}  [checkpointed]")
    fout.close()
    print(f"done; +{n_new} new rows, {len(done)} total -> {OUTCSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
