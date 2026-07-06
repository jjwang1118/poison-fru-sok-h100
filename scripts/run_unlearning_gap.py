#!/usr/bin/env python3
"""Driver for the Path-A algorithm-level gap experiment (TARGETED / ER@K axis).

It orchestrates the (patched) CFRU CLI and parses each run's
`fedtasksave/<task>/<exp>/record/history{R}.pkl` to build per-seed rows in
`data/unlearning_gap.csv`.

Attack = psmu (Route A: PSMU synthetic-user model-poisoning; acts on the uploaded/
aggregated model so it reaches the ER path). Override with ATTACK=targetPromo|fedAttack.
Primary metric = ER@K (exposure of the target item). HR/NDCG (utility) are also
recorded so utility-vs-exposure can be contrasted (utility ~ unchanged while exposure
differs is the headline). Tune PSMU_SCALE (env) upward if ER(poison) stays below the gate.

Per (seed, forget_frac) it launches THREE runs:
  clean   : --atk_method none        --forget_frac 0  --mode unlearn
              -> accuracy / er             = HR,NDCG / ER@K of M_clean
  unlearn : --atk_method targetPromo  --forget_frac f  --mode unlearn
              -> accuracy / er             = M_poison ;  accuracy_unlearn / er_unlearn = M^{-F}
  retrain : --atk_method targetPromo  --forget_frac f  --mode retrain
              -> accuracy_retain / er_retain = M_retain

Run it FROM inside external/CFRU (so relative paths match the repo):
    cd external/CFRU && python3 ../../scripts/run_unlearning_gap.py
Smoke test first: set SEEDS=[0], FORGET_FRACS=[1.0]; check ER(poison) >> ER(clean).
"""
from __future__ import annotations

import csv, io, math, os, pickle, subprocess, sys
from pathlib import Path

# ---- experiment grid (env-overridable; smoke.sh sets a small subset) --------
# Any of these can be overridden by an environment variable of the same name, e.g.
#   SEEDS=0 FORGET_FRACS=1.0 NUM_ROUNDS=20 BATCH_SIZE=512 python3 .../run_unlearning_gap.py
def _envstr(k, d):  return os.environ.get(k, d)
def _envint(k, d):  return int(os.environ.get(k, d))
def _envflt(k, d):  return float(os.environ.get(k, d))
def _envnums(k, d):
    v = os.environ.get(k)
    if not v:
        return d
    return [float(x) if "." in x else int(x) for x in v.replace(" ", "").split(",") if x]

ATTACK        = _envstr("ATTACK", "psmu")            # psmu (Route A) | targetPromo | fedAttack | fedFlipGrads
ANCHOR        = _envstr("ANCHOR", f"CFRU-{ATTACK}-ML1M-NCF")
TASK          = "movielens1m_cnum100_dist0_skew0_seed0"
MODEL         = "NCF"
NUM_ROUNDS    = _envint("NUM_ROUNDS", 40)            # appears in the exp folder name
TOPN          = _envint("TOPN", 20)                  # HR@K / NDCG@K / ER@K
S1            = 8                                     # repo's S1 (appears in the exp folder name)
PROPORTION    = 1.0                                  # sample() uses ALL clients (full participation)
ALPHA         = 0.5
LEARNING_RATE = _envflt("LEARNING_RATE", 0.01)       # repo default 0.1 collapses NCF; 0.01 is healthy
NUM_EPOCHS    = _envint("NUM_EPOCHS", 5)
BATCH_SIZE    = _envint("BATCH_SIZE", 64)            # raise (256/512) for GPU throughput
NUM_NG        = _envint("NUM_NG", 4)                 # negatives per positive
SEEDS         = _envnums("SEEDS", [0, 1, 2, 3, 4, 5])           # comma list, e.g. "0" or "0,1,2"
FORGET_FRACS  = _envnums("FORGET_FRACS", [0.2, 0.4, 0.6, 0.8, 1.0])  # e.g. "1.0" or "0.2,0.6,1.0"
GPU_ID        = _envint("GPU_ID", 0)                 # --server_gpu_id; CFRU auto-uses cuda:GPU_ID when available
TARGET_ITEM   = _envint("TARGET_ITEM", -1)           # -1 = CFRU auto-picks the least-popular item (clean ER ~ 0)
PSMU_SCALE    = _envflt("PSMU_SCALE", 5.0)           # amplify crafted target shift vs FedAvg dilution [tune for ER gate]
PSMU_S        = _envint("PSMU_S", 8)                 # synthetic users per malicious client per round
PY            = sys.executable

REPO   = Path(__file__).resolve().parents[1] / "external" / "CFRU"
OUTCSV = Path(_envstr("OUTCSV", str(Path(__file__).resolve().parents[1] / "data" / "unlearning_gap.csv")))


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
           "--learning_rate", str(LEARNING_RATE), "--server_gpu_id", str(GPU_ID),
           "--target_item", str(TARGET_ITEM),
           "--psmu_scale", str(PSMU_SCALE), "--psmu_s", str(PSMU_S)]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)
    pkl = exp_dir(seed, atk) / f"history{NUM_ROUNDS}.pkl"
    with open(pkl, "rb") as f:
        return _CPU_Unpickler(f).load()      # dict: accuracy[/_unlearn/_retain], er[/_unlearn/_retain], target_item


def load_pkl(seed: int, atk: str):
    """Reuse an existing run's result from disk (used to skip re-running a clean run)."""
    pkl = exp_dir(seed, atk) / f"history{NUM_ROUNDS}.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            return _CPU_Unpickler(f).load()
    return None


FIELDS = ["anchor", "method", "seed", "forget_frac", "attack_saturated", "target_item",
          "hr_clean", "hr_poison", "hr_retain", "hr_unlearned",
          "ndcg_clean", "ndcg_poison", "ndcg_retain", "ndcg_unlearned",
          "er_clean", "er_poison", "er_retain", "er_unlearned"]
ATTACK_ER_MIN = 0.30   # 'saturated' = ER@K of the poisoned model >= this (target promoted; tune from smoke)


def _g(rec, key, default=float("nan")):
    v = rec.get(key, default) if isinstance(rec, dict) else default
    return default if v is None else v


def _read_existing():
    """Resume support: {(seed,frac): row}, {seed: (hr,ndcg,er,target)}, header_ok."""
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
                clean[s] = (float(r["hr_clean"]), float(r["ndcg_clean"]),
                            float(r["er_clean"]), r.get("target_item", ""))
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
        # clean once per seed: CSV cache -> on-disk pkl (must already carry ER) -> run
        if seed in clean_cache:
            hr_c, ndcg_c, er_c, tgt = clean_cache[seed]
            print(f"  [skip]  seed={seed} clean (from CSV): HR={hr_c:.4f} ER={er_c:.4f}")
        else:
            rec = load_pkl(seed, "none")
            if not (rec and "accuracy" in rec and "er" in rec):
                rec = run(seed, "none", 0.0, "unlearn")
            hr_c, ndcg_c = rec["accuracy"]
            er_c, tgt = _g(rec, "er"), _g(rec, "target_item", "")
            clean_cache[seed] = (hr_c, ndcg_c, er_c, tgt)
            print(f"  [clean] seed={seed}: HR={hr_c:.4f} ER={er_c:.4f} target_item={tgt}")

        for frac in FORGET_FRACS:
            if (seed, frac) in done:
                print(f"  [skip]  seed={seed} frac={frac} (already in CSV)")
                continue
            ru = run(seed, ATTACK, frac, "unlearn")        # M_poison + M^{-F}
            rr = run(seed, ATTACK, frac, "retrain")        # M_retain
            hr_p, ndcg_p = ru["accuracy"]
            hr_u, ndcg_u = ru["accuracy_unlearn"]
            hr_r, ndcg_r = rr["accuracy_retain"]
            er_p, er_u, er_r = _g(ru, "er"), _g(ru, "er_unlearn"), _g(rr, "er_retain")
            sat = 0 if (isinstance(er_p, float) and math.isnan(er_p)) else int(er_p >= ATTACK_ER_MIN)
            row = {
                "anchor": ANCHOR, "method": "CFRU", "seed": seed, "forget_frac": frac,
                "attack_saturated": sat, "target_item": _g(ru, "target_item", tgt),
                "hr_clean": hr_c, "hr_poison": hr_p, "hr_retain": hr_r, "hr_unlearned": hr_u,
                "ndcg_clean": ndcg_c, "ndcg_poison": ndcg_p, "ndcg_retain": ndcg_r, "ndcg_unlearned": ndcg_u,
                "er_clean": er_c, "er_poison": er_p, "er_retain": er_r, "er_unlearned": er_u,
            }
            writer.writerow(row); fout.flush()                  # checkpoint after each (seed,frac)
            done[(seed, frac)] = row; n_new += 1
            print(f"  -> seed={seed} frac={frac}: ER clean={er_c:.3f} poison={er_p:.3f} "
                  f"retain={er_r:.3f} unlearn={er_u:.3f} | HR clean={hr_c:.3f} poison={hr_p:.3f}  [checkpointed]")
    fout.close()
    print(f"done; +{n_new} new rows, {len(done)} total -> {OUTCSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
