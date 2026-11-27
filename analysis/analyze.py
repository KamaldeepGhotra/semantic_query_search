"""Compute every quantitative claim of ThalamusDB (SIGMOD'24) Sec. 7 from our logs.

Usage: python analysis/analyze.py [--dbs craigslist netflix]
Writes analysis/results.md (paper vs reproduced) and analysis/*.png figures.
"""

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tdb"))
from constraint import TDBConstraint  # noqa: E402

LOGS, OUT = ROOT / "logs", ROOT / "analysis"
CFGS = ["Cl", "cL", "Ce", "cE", "Le", "lE"]
CONSTRAINTS = {"Cl": ("error", 0.1, 1), "cL": ("error", 0.1, 10000), "Le": ("runtime", 10, 1),
               "lE": ("runtime", 10, 10000), "Ce": ("feedback", 5, 1), "cE": ("feedback", 5, 10000)}
TEXT_COLS = {"title_u", "featured_review_u", "description_u"}
DB_TAG = {"craigslist": "Cr", "netflix": "N", "youtubeaudios": "Y"}


def load(name):
    p = LOGS / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def agg_type(sql):
    s = sql.lower()
    if " limit " in s:
        return "Limit"
    for a, n in [("count(", "Cnt"), ("min(", "Min"), ("max(", "Max"), ("sum(", "Sum"), ("avg(", "Avg")]:
        if s.startswith(f"select {a}"):
            return n
    return "?"


def nl_cols(sql):
    import re
    return [m.split(",")[0].strip() for m in re.findall(r"nl\((.+?), '", sql)]


def cost(cfg, error, feedback, runtime):
    return TDBConstraint(*CONSTRAINTS[cfg]).cost(error, feedback, runtime)


def exact_value(lus):
    lus = ast.literal_eval(lus) if isinstance(lus, str) else None
    return None if not lus else lus[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbs", nargs="+", default=["youtubeaudios", "craigslist", "netflix"])
    args = ap.parse_args()
    lines = ["# ThalamusDB reproduction: paper vs ours\n",
             "Models: Gemma 3 4B (text, images) and Gemma 3n E4B (audio) replace Sentence-BERT/CLIP/the audio retrieval model. "
             "Simulated labels use Gemma's Yes/No boundary (score >= 0). "
             "Runtime = measured Gemma latency (M5 Pro) + real SQL/optimizer time.\n"]
    main_runs = {db: load(f"main_{db}") for db in args.dbs}
    main_runs = {db: d for db, d in main_runs.items() if d is not None}
    for db, d in main_runs.items():
        d["agg"] = d["sql"].map(agg_type)

    # ---------- Fig. 7: speedup vs exact processing (MindsDB stand-in), text-only queries ----------
    sp = []
    for db, d in main_runs.items():
        tdb = d[(d.method == "local") & (d.config == "Cl")]
        ex = d[d.method == "naive"].set_index("sql")
        for _, r in tdb.iterrows():
            if set(nl_cols(r.sql)) <= TEXT_COLS and r.sql in ex.index:
                sp.append({"db": db, "sql": r.sql, "speedup": ex.loc[r.sql, "total_time"] / r.total_time})
    sp = pd.DataFrame(sp)
    lines.append("\n## Fig. 7 — speedup over exact processing (text-predicate queries, error <= 10%)\n")
    lines.append("Paper compares to MindsDB (exact, given true thresholds); we use the Exact baseline, which is the same work.\n")
    if len(sp):
        lines.append(f"| | Paper | Ours |\n|---|---|---|\n| Queries | 12 (incl. YouTube) | {len(sp)} |\n"
                     f"| Average speedup | 35.0x | {sp.speedup.mean():.1f}x |\n| Max speedup | 1,648.4x | {sp.speedup.max():.1f}x |\n")
        for db, g in sp.groupby("db"):
            lines.append(f"- {db}: mean {g.speedup.mean():.1f}x over {len(g)} queries\n")

    # ---------- Fig. 9: ablations, scaled cost vs ThalamusDB ----------
    lines.append("\n## Fig. 9 — ablations, cost relative to ThalamusDB (=1)\n")
    lines.append("| Benchmark | Config | Exact | +Ordered | +Approx |\n|---|---|---|---|---|\n")
    abl = []
    for db, d in main_runs.items():
        for cfg in CFGS:
            tdb = d[(d.method == "local") & (d.config == cfg)].set_index("sql")
            if tdb.empty:
                continue
            tdb_cost = tdb.apply(lambda r: cost(cfg, r.error, r.nr_feedbacks, r.total_time), axis=1)
            row = {"db": db, "cfg": cfg}
            for m, name in [("naive", "Exact"), ("ordered", "+Ordered"), ("random", "+Approx")]:
                sub = d[d.method == m] if m != "random" else d[(d.method == m) & (d.config == cfg)]
                if sub.empty:
                    continue
                c = sub.apply(lambda r: cost(cfg, r.error, r.nr_feedbacks, r.total_time), axis=1).groupby(sub.sql).min()
                common = c.index.intersection(tdb_cost.index)
                row[name] = float((c[common] / tdb_cost[common]).mean())
            abl.append(row)
            lines.append(f"| {db} | {cfg} | {row.get('Exact', float('nan')):.1f} | {row.get('+Ordered', float('nan')):.1f} | {row.get('+Approx', float('nan')):.1f} |\n")
    abl = pd.DataFrame(abl)
    if len(abl):
        lines.append(f"\n| Average overhead | Paper | Ours |\n|---|---|---|\n"
                     f"| Exact (all benchmarks) | 951.5x | {abl['Exact'].mean():.1f}x |\n"
                     f"| +Ordered (all benchmarks) | 945.5x | {abl['+Ordered'].mean():.1f}x |\n")
        paper_approx = {"craigslist": "162.1x", "netflix": "6.0x", "youtubeaudios": "34.0x"}
        for db, g in abl.groupby("db"):
            lines.append(f"| +Approx ({db}) | {paper_approx[db]} | {g['+Approx'].mean():.1f}x |\n")

    # ---------- Fig. 10: runtime breakdown ----------
    lines.append("\n## Fig. 10 — runtime breakdown of ThalamusDB (average over configs and queries)\n")
    lines.append("| Benchmark | Model inference (paper) | Model inference (ours) | Relational (paper) | Relational (ours) | Optimization (ours) |\n|---|---|---|---|---|---|\n")
    paper_bd = {"craigslist": (89.6, 2.5), "netflix": (50.7, 45.5), "youtubeaudios": (92.4, 4.1)}
    for db, d in main_runs.items():
        t = d[d.method == "local"]
        ml, sql, opt = (t.time_ml / t.total_time).mean() * 100, (t.time_sql / t.total_time).mean() * 100, (t.time_optimize / t.total_time).mean() * 100
        lines.append(f"| {db} | {paper_bd[db][0]}% | {ml:.1f}% | {paper_bd[db][1]}% | {sql:.1f}% | {opt:.1f}% |\n")

    # ---------- Fig. 12/13: error analysis ----------
    lines.append("\n## Fig. 12 / 13 — bound error and relative error\n")
    lines.append("Relative error = |true - midpoint(bounds)| / true; true value from the Exact baseline.\n\n")
    lines.append("| Benchmark | Config | max bound error | median bound error | median relative error | 80th pct relative error |\n|---|---|---|---|---|---|\n")
    err_rows = []
    for db, d in main_runs.items():
        truth = d[d.method == "naive"].set_index("sql")["lus"].map(exact_value)
        for cfg in CFGS:
            t = d[(d.method == "local") & (d.config == cfg) & (d["agg"] != "Limit")]
            rel = []
            for _, r in t.iterrows():
                lu = ast.literal_eval(r.lus)[0]
                vt = truth.get(r.sql)
                if vt is None or not np.isfinite(lu[0]) or not np.isfinite(lu[1]):
                    rel.append(1.0)
                    continue
                ve = (lu[0] + lu[1]) / 2
                rel.append(0.0 if vt == ve else abs(vt - ve) / abs(vt) if vt else 1.0)
            t_all = d[(d.method == "local") & (d.config == cfg)]
            if t_all.empty:
                continue
            err_rows.append({"db": db, "cfg": cfg, "rel": rel, "bound": t_all.error.tolist()})
            lines.append(f"| {db} | {cfg} | {t_all.error.max():.3f} | {t_all.error.median():.3f} | "
                         f"{np.median(rel):.3f} | {np.percentile(rel, 80):.3f} |\n")
    lines.append("\nPaper reference points: cE median relative error 5.2% (YouTube), 3.9% (Craigslist) vs Ce 100%; "
                 "Netflix cE 80th pct 0.8% vs Ce 44.4%. Bound error <= 10% for every query under Cl and cL.\n")
    lines.append("\n**Fig. 13 — mean bound error by aggregate type (error <= 10%)**\n\n| Benchmark | Config | Min | Max | Sum | Avg | Cnt | Limit |\n|---|---|---|---|---|---|---|---|\n")
    for db, d in main_runs.items():
        for cfg in ["Cl", "cL"]:
            t = d[(d.method == "local") & (d.config == cfg)]
            m = t.groupby("agg").error.mean()
            lines.append(f"| {db} | {cfg} | " + " | ".join(f"{m.get(a, float('nan')):.3f}" for a in ["Min", "Max", "Sum", "Avg", "Cnt", "Limit"]) + " |\n")

    # ---------- Fig. 14: F1 of limit queries ----------
    lines.append("\n## Fig. 14 — F1 of LIMIT queries\n\n| Benchmark | " + " | ".join(CFGS) + " |\n" + "|---" * (len(CFGS) + 1) + "|\n")
    for db, d in main_runs.items():
        t = d[(d.method == "local") & (d["agg"] == "Limit")]
        m = t.groupby("config").f1.mean()
        lines.append(f"| {db} | " + " | ".join(f"{m.get(c, float('nan')):.2f}" for c in CFGS) + " |\n")
    lines.append("\nPaper: highest F1 under Cl/cL, then cE/lE, lowest under Ce/Le.\n")

    # ---------- Table 5: post-processing ----------
    lines.append("\n## Table 5 — post-processing (error <= 10%)\n\n| Benchmark | Config | #post / #all actions (ours) | error improvement (ours) |\n|---|---|---|---|\n")
    for db, d in main_runs.items():
        for cfg in ["Cl", "cL"]:
            t = d[(d.method == "local") & (d.config == cfg)]
            if t.empty or "nr_post_actions" not in t:
                continue
            ratio = t.nr_post_actions.sum() / max(t.nr_actions.sum(), 1)
            imp = (t.error_before_post.fillna(t.error) - t.error).mean()
            lines.append(f"| {db} | {cfg} | {ratio:.3f} | {imp:.3f} |\n")
    lines.append("\nPaper: YouTube Cl 0.183 / 0.153, cL 0.248 / 0.246; Craigslist Cl 0.145 / 0.254, cL 0.170 / 0.247; Netflix Cl 0.007 / 0.001, cL 0.000 / 0.000.\n")

    # ---------- Fig. 8: ABAE ----------
    lines.append("\n## Fig. 8 — ABAE vs ThalamusDB (Count/Sum/Avg; error 5/10/50%; weight 1..1e4)\n")
    comp = []
    for db in args.dbs:
        f8, ab, d = load(f"fig8_{db}"), load(f"abae_{db}"), main_runs.get(db)
        if f8 is None or ab is None or d is None:
            continue
        exact_fb = d[d.method == "naive"].set_index("sql").nr_feedbacks
        f8["agg"] = f8.sql.map(agg_type)
        for _, r in f8.iterrows():
            cols = nl_cols(r.sql)
            pre = "youtube" if db == "youtubeaudios" else db
            if db == "netflix":
                case = "netflix_p"
            elif " and nl(" in r.sql:
                case = f"{pre}_and"
            elif " or nl(" in r.sql:
                case = f"{pre}_or"
            elif db == "youtubeaudios":
                case = "youtube_a" if cols == ["audio"] else "youtube_d"
            else:
                case = "craigslist_b" if cols == ["img"] else "craigslist_w"
            target = float(r.config.split("_")[0][1:])
            w = float(r.config.split("_w")[1])
            a = ab[(ab.case == case) & (ab["agg"] == {"Cnt": "count", "Sum": "sum", "Avg": "avg"}[r["agg"]]) & np.isclose(ab.target, target)]
            if a.empty:
                continue
            a = a.iloc[0]
            tdb_cost = r.total_time + w * r.nr_feedbacks
            for variant, labels in [("ABAEmin", 3 * a.nr_preds), ("ABAEmax", exact_fb.get(r.sql, np.nan))]:
                abae_cost = a.time + w * labels
                comp.append({"db": db, "case": case, "agg": r["agg"], "target": target, "w": w, "variant": variant,
                             "failed": bool(a.failed), "ratio": abae_cost / tdb_cost,
                             "tdb_wins": bool(a.failed) or abae_cost > tdb_cost})
    comp = pd.DataFrame(comp)
    if len(comp):
        lines.append("| | Paper | Ours |\n|---|---|---|\n")
        for v, pw, pr in [("ABAEmax", "91.1%", "2.4x"), ("ABAEmin", "67.7%", "2.1x")]:
            c = comp[comp.variant == v]
            lines.append(f"| ThalamusDB cheaper than {v} | {pw} | {100 * c.tdb_wins.mean():.1f}% |\n"
                         f"| Avg cost overhead of {v} (excl. ABAE failures) | {pr} | {c[~c.failed].ratio.mean():.1f}x |\n")
        lines.append(f"| ThalamusDB beats ABAE, all cases (abstract) | 78.9% | {100 * comp.tdb_wins.mean():.1f}% |\n")
        lines.append(f"| ABAE failed to meet error bound | — | {100 * comp.drop_duplicates(['case', 'agg', 'target']).failed.mean():.1f}% of (query, bound) |\n")
        by_t = comp.groupby("target").tdb_wins.mean() * 100
        lines.append("\nThalamusDB win rate by error bound: " + ", ".join(f"{t:g}: {v:.1f}%" for t, v in by_t.items()) +
                     " (paper: margin grows as the bound tightens).\n")

    # ---------- Fig. 11: selectivity ----------
    f11 = load("fig11_craigslist")
    if f11 is not None:
        f11["agg"] = f11.sql.map(agg_type)
        f11["pred"] = f11.sql.str.extract(r"nl\(img, '(.+?)'\)")[0]
        sel = {}
        for p in f11.pred.unique():
            s = pd.read_parquet(ROOT / "data/scores" / f"craigslist_img_{p.replace(' ', '_')}.parquet")
            sel[p] = float((s.score >= 0).mean())
        f11["cost"] = f11.total_time + f11.nr_feedbacks
        piv = f11.pivot_table(index=["agg", "pred"], columns="method", values="cost")
        piv["ratio"] = piv["naive"] / piv["local"]
        lines.append("\n## Fig. 11 — ThalamusDB vs Exact by predicate selectivity (Craigslist images, error <= 10%, w = 1)\n\n")
        lines.append("| Predicate | Selectivity | " + " | ".join(["Min", "Max", "Sum", "Avg", "Cnt", "Limit"]) + " |\n" + "|---" * 8 + "|\n")
        for p in sorted(sel, key=sel.get):
            vals = [piv["ratio"].get((a, p), float("nan")) for a in ["Min", "Max", "Sum", "Avg", "Cnt", "Limit"]]
            lines.append(f"| {p} | {sel[p]:.3f} | " + " | ".join(f"{v:.1f}x" for v in vals) + " |\n")
        lines.append("\nCells = Exact cost / ThalamusDB cost. Paper: large gains for Min/Max/Limit, smaller for Count/Avg, "
                     "and a tendency toward larger gains for less selective predicates.\n")

    (OUT / "results.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
