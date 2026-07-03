#!/usr/bin/env python3
"""Render the Path-A (utility-degradation / UDR) algorithm-level gap table.

Input: data/unlearning_gap.csv with ONE ROW PER SEED:

    anchor,method,seed,forget_frac,attack_saturated,
    hr_clean,hr_poison,hr_retain,hr_unlearned,
    ndcg_clean,ndcg_poison,ndcg_retain,ndcg_unlearned

Effect = utility (HR@K). Gap decomposition on means over saturated seeds:
    id_gap    = HR(clean) - HR(retain)      (carrier identification)
    au_gap    = HR(retain) - HR(unlearned)  (the unlearning ALGORITHM)
    total_gap = HR(clean) - HR(unlearned)   (= id_gap + au_gap)
    UDR       = total_gap / (HR(clean) - HR(poison))   (normalized)

With no saturated rows it writes a comment-only stub (no [CHECK]); section 8 then
reads as a *protocol*, not a reported result.
"""
from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "unlearning_gap.csv"
TEX = ROOT / "paper" / "tables" / "unlearning_gap.tex"
TRUE = {"1", "yes", "true", "y", "t"}


def s(x: float) -> str:
    return f"{x:+.4f}"


def main() -> int:
    raw = [r for r in csv.DictReader(CSV.open(newline="", encoding="utf-8"))
           if r.get("anchor", "").strip()]
    rows = [r for r in raw if r.get("attack_saturated", "").strip().lower() in TRUE]

    if not rows:
        TEX.write_text(
            "% Algorithm-level utility-gap table: no data yet.\n"
            "% Run scripts/run_unlearning_gap.py (CFRU + fedAttack), then\n"
            "% python3 scripts/make_unlearning_gap_table.py, then \\input this file in section 8.\n",
            encoding="utf-8")
        print("Wrote comment-only stub (no saturated rows yet).")
        return 0

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["anchor"], r["method"], r["forget_frac"])].append(r)

    def m(g, k):
        return st.mean(float(x[k]) for x in g)

    body = []
    for (anchor, method, frac), g in sorted(groups.items()):
        n = len(g)
        hc, hp, hr, hu = m(g, "hr_clean"), m(g, "hr_poison"), m(g, "hr_retain"), m(g, "hr_unlearned")
        idg, aug, tot = hc - hr, hr - hu, hc - hu
        denom = hc - hp
        udr = (tot / denom) if abs(denom) > 1e-9 else float("nan")
        au_sd = st.pstdev([float(x["hr_retain"]) - float(x["hr_unlearned"]) for x in g]) if n > 1 else 0.0
        udr_str = "n/a" if udr != udr else f"{udr:+.3f}"
        body.append(
            f"{anchor} & {method} & {frac} & {n} & {hc:.4f} & {hp:.4f} & "
            f"{s(idg)} & {s(aug)}\\,($\\pm${au_sd:.4f}) & {s(tot)} & {udr_str} \\\\")

    caption = (
        r"\caption{Algorithm-level \emph{utility-degradation} gap for official CFRU under the "
        r"untargeted \textsf{fedAttack}, with nested forget sets $F_1\subset\cdots\subset F_k\subseteq C$ "
        r"on a fixed poisoned run, over $n$ attack-saturated seeds. Effect is utility ($\mathrm{HR@20}$). "
        r"\emph{id-gap}$=\mathrm{HR}(M_{\mathsf{clean}})-\mathrm{HR}(M_{\mathsf{retain}})$ "
        r"(incomplete carrier identification); \emph{au-gap}"
        r"$=\mathrm{HR}(M_{\mathsf{retain}})-\mathrm{HR}(M^{-F})$ (the unlearning algorithm itself); "
        r"\emph{total}$=$id-gap$+$au-gap; \emph{UDR}$=$total$/(\mathrm{HR}(M_{\mathsf{clean}})-"
        r"\mathrm{HR}(M_{\mathsf{poison}}))$. A positive au-gap means the unlearned model is worse in "
        r"utility than its own retrain target. This is an untargeted-attack table; it is reported "
        r"separately from the target-promotion (ER@K) case study.}")
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\footnotesize", caption,
        r"\label{tab:unlearning-gap}",
        r"\begin{tabular}{@{}lllccccccc@{}}", r"\toprule",
        r"Anchor & Method & $|F|/|C|$ & $n$ & $\mathrm{HR}_{\mathrm{clean}}$ & "
        r"$\mathrm{HR}_{\mathrm{poison}}$ & id-gap & au-gap & total & UDR \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    TEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {TEX.relative_to(ROOT)} ({len(groups)} groups from {len(rows)} saturated seed-rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
