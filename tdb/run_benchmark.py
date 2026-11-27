"""Benchmark driver mirroring the authors' benchmark.main(), with Gemma score caches.

Usage:
  python run_benchmark.py --dbs craigslist youtubeaudios netflix --methods local naive ordered random
Methods (authors' names): local = ThalamusDB, naive = Exact, ordered = +Ordered, random = +Approx.
Writes one CSV row per (constraint, db, method, query[, permutation]) to logs/<out>.csv.
"""

import argparse
import datetime
import itertools
import random
import sys
import time
from pathlib import Path

import pandas as pd

import simclock

simclock.install()  # Before other imports capture time.time.

import nldbs  # noqa: E402
from benchmark import queries_craigslist, queries_netflix, queries_youtubeaudios  # noqa: E402
from constraint import TDBConstraint  # noqa: E402
from query import NLQuery  # noqa: E402

# Simulated user labels: an item satisfies a predicate iff Gemma's Yes-No logit gap >= 0.
# (The paper used manually determined per-predicate thresholds on CLIP/BERT scores.)
GROUND_TRUTHS = {p: 0.0 for col in nldbs.CACHE_TASKS.values() for p in col}

# Paper Sec. 7.3: Cl, cL, Ce, cE, Le, lE  (constrained metric, bound, weight).
CONSTRAINTS = {
    "Cl": ("error", 0.1, 1), "cL": ("error", 0.1, 10000),
    "Le": ("runtime", 10, 1), "lE": ("runtime", 10, 10000),
    "Ce": ("feedback", 5, 1), "cE": ("feedback", 5, 10000),
}
# Fig. 8: error constraint in {5%, 10%, 50%} x weight in {1, 10, 1e2, 1e3, 1e4}, Count/Sum/Avg queries.
for _e in (0.05, 0.1, 0.5):
    for _w in (1, 10, 100, 1000, 10000):
        CONSTRAINTS[f"e{_e}_w{_w}"] = ("error", _e, _w)
FIG8_AGGS = ("count(", "sum(", "avg(")
QUERIES = {"craigslist": queries_craigslist, "youtubeaudios": queries_youtubeaudios, "netflix": queries_netflix}
ROOT = Path(__file__).resolve().parent.parent


def run_one(dbname, method, sql, constraint, permutation):
    query = NLQuery(sql)
    nldb = nldbs.get_nldb_by_name(dbname)
    start = time.time()
    if method == "naive":
        info = nldb.run_baseline(query, GROUND_TRUTHS)
    elif method == "ordered":
        info = nldb.run_baseline_ordered(query, permutation, GROUND_TRUTHS)
    else:
        info = nldb.run(query, constraint, GROUND_TRUTHS, method)
    total = time.time() - start
    info["f1"] = limit_f1(nldb, query, info)
    return info, total


def limit_f1(nldb, query, info):
    """F1 of a LIMIT query's returned rows (definitely-true set) against ground truth.
    Precision = returned rows that truly satisfy the predicates / returned rows;
    recall = truly satisfying returned rows / LIMIT (every benchmark predicate has > LIMIT matches)."""
    rows = info.get("rows_l")
    if query.limit < 0 or rows is None:
        return None
    cols = info["cols_l"]
    is_or = " or " in query.sql
    nr_true = 0
    for row in rows:
        hits = [nldb.get_col_by_name(c).processor.caches[t][0][row[cols.index(c)]] >= GROUND_TRUTHS[t]
                for c, t in query.nl_preds]
        nr_true += any(hits) if is_or else all(hits)
    precision = nr_true / len(rows) if rows else 0.0
    recall = nr_true / query.limit
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbs", nargs="+", default=list(QUERIES))
    ap.add_argument("--methods", nargs="+", default=["local"])
    ap.add_argument("--configs", nargs="+", default=list(CONSTRAINTS)[:6])
    ap.add_argument("--out", default="benchmark")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--fig8", action="store_true", help="only Count/Sum/Avg queries; configs default to Fig. 8 grid")
    ap.add_argument("--fig11", action="store_true", help="Craigslist single image-predicate queries, 7 predicates")
    args = ap.parse_args()
    if args.fig8 and args.configs == list(CONSTRAINTS)[:6]:
        args.configs = [c for c in CONSTRAINTS if c.startswith("e")]
    random.seed(1)

    out = ROOT / "logs" / f"{args.out}.csv"
    out.parent.mkdir(exist_ok=True)
    for cfg in args.configs:
        constraint = TDBConstraint(*CONSTRAINTS[cfg])
        for dbname in args.dbs:
            sqls = QUERIES[dbname]()[: args.max_queries]
            if args.fig11:
                # Fig. 11: single NL predicate on images.img, each aggregate type plus Limit.
                preds = ["leather sofa", "blue chair", "wooden chair", "dining table", "sofa", "table", "wooden"]
                aggs = ["min(price)", "max(price)", "sum(price)", "avg(price)", "count(*)"]
                sqls = [f"select {a} from images, furniture where nl(img, '{p}') and images.aid = furniture.aid" for p in preds for a in aggs]
                sqls += [f"select * from images, furniture where nl(img, '{p}') and images.aid = furniture.aid limit 10" for p in preds]
            if args.fig8:
                sqls = [s for s in sqls if " limit " not in s and any(a in s for a in FIG8_AGGS)]
            for method in args.methods:
                # Exact / +Ordered ignore constraints: run them only once (first config).
                if method in ("naive", "ordered") and cfg != args.configs[0]:
                    continue
                for sql in sqls:
                    n = len(NLQuery(sql).nl_preds)
                    perms = list(itertools.permutations(range(n))) if method == "ordered" else [None]
                    for perm in perms:
                        info, total_time = run_one(dbname, method, sql, constraint, perm)
                        row = {
                            "timestamp": str(datetime.datetime.now()), "db": dbname, "method": method,
                            "config": cfg, "constraint": str(constraint), "sql": sql, "permutation": str(perm),
                            "total_time": total_time, "time_optimize": info.get("time_optimize"),
                            "time_sql": info.get("time_sql"), "time_ml": info.get("time_ml"),
                            "error": info["error"], "lus": str(info.get("lus")),
                            "percent_processed": str(info["processed"]), "nr_feedbacks": sum(info["feedback"]),
                            "nr_actions": info.get("nr_actions"), "f1": info.get("f1"), "nr_post_actions": info.get("nr_post_actions", 0), "error_before_post": info.get("error_before_post"),
                            "cost": constraint.cost(info["error"], sum(info["feedback"]), total_time),
                        }
                        print(f"RESULT {cfg} {dbname} {method} t={total_time:.1f}s err={info['error']:.3f} | {sql}", flush=True)
                        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    sys.exit(main())
