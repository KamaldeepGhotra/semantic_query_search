"""ABAE baseline (Kang et al., VLDB'21) adapted as in ThalamusDB Sec. 7.2 / Fig. 8.

ABAE's released code (stanford-futuredata/abae) only covers its own datasets; ThalamusDB's
adapter was not released. This reimplements ABAE's two-stage stratified sampling
(abae/algorithm.py::_execute_ours: K strata by proxy score, n1 = n*C/K per stratum in stage 1,
n2 = n*(1-C) allocated proportional to sqrt(p_k)*sigma_k in stage 2) with these settings
from the ThalamusDB paper:
  * proxy scores = the oracle model's own scores, free of charge;
  * NL-predicate labels provided free of computation; ABAEmin is charged 3 labels per
    predicate, ABAEmax the number of labels exact binary search needs (Exact baseline);
  * confidence 1 - 1e-9. With that confidence a bootstrap percentile is not resolvable,
    so the interval uses a normal approximation (z = 6.109) on the stratified estimator.
  * Sample size grows until bound error (u-l)/(u+l) <= target; if all rows are sampled
    without meeting it, the run is marked failed (paper: darkest shade in Fig. 8).
Computation cost = summed Gemma latency of distinct items whose predicate was evaluated.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import nldbs

ROOT = Path(__file__).resolve().parent.parent
K, C, Z = 5, 0.5, 6.109  # z for two-sided 1 - 1e-9
GT = 0.0  # Ground-truth threshold on the Yes-No logit gap.


def _cache(task):
    df = pd.read_parquet(nldbs.SCORES / f"{task}.parquet").sort_values("idx")
    return df["score"].to_numpy(), df["latency"].to_numpy()


def build_rows(case):
    """Return dict with per-row arrays: proxy, pred, value, and item ids/latencies per predicate."""
    if case.startswith("craigslist"):
        imgs = nldbs.get_df_by_name("furniture_imgs")
        furn = nldbs.get_df_by_name("furniture")
        aid_to_f = pd.Series(np.arange(len(furn)), index=furn["aid"])
        f_idx = aid_to_f.loc[imgs["aid"]].to_numpy()
        i_idx = np.arange(len(imgs))
        value = furn["price"].to_numpy()[f_idx].astype(float)
        s_img, l_img = _cache("craigslist_img_blue_chair")
        s_tit, l_tit = _cache("craigslist_title_wood")
        preds = {"b": [(s_img, l_img, i_idx)], "w": [(s_tit, l_tit, f_idx)],
                 "and": [(s_img, l_img, i_idx), (s_tit, l_tit, f_idx)],
                 "or": [(s_img, l_img, i_idx), (s_tit, l_tit, f_idx)]}[case.split("_")[1]]
        combine = "or" if case.endswith("or") else "and"
    elif case.startswith("youtube"):
        yt = nldbs.get_df_by_name("youtube")
        r_idx = np.arange(len(yt))
        value = yt["likes"].to_numpy().astype(float)
        s_aud, l_aud = _cache("youtube_audio_voices")
        s_dsc, l_dsc = _cache("youtube_description_cooking")
        preds = {"a": [(s_aud, l_aud, r_idx)], "d": [(s_dsc, l_dsc, r_idx)],
                 "and": [(s_aud, l_aud, r_idx), (s_dsc, l_dsc, r_idx)],
                 "or": [(s_aud, l_aud, r_idx), (s_dsc, l_dsc, r_idx)]}[case.split("_")[1]]
        combine = "or" if case.endswith("or") else "and"
    elif case == "netflix_p":
        movies = nldbs.get_df_by_name("movies")
        ratings = nldbs.get_df_by_name("ratings")
        mid_to_m = pd.Series(np.arange(len(movies)), index=movies["movieid"])
        m_idx = mid_to_m.loc[ratings["movieid"]].to_numpy()
        value = ratings["rating"].to_numpy().astype(float)
        s, l = _cache("netflix_review_positive")
        preds, combine = [(s, l, m_idx)], "and"
    else:
        raise ValueError(case)
    row_scores = np.stack([s[idx] for s, _, idx in preds])
    truth = (row_scores >= GT)
    pred = truth.all(0) if combine == "and" else truth.any(0)
    proxy = row_scores.min(0) if combine == "and" else row_scores.max(0)
    return {"proxy": proxy, "pred": pred, "value": value, "preds": preds}


def abae_estimate(rows, agg, n, rng):
    """One ABAE run with total budget n. Returns (estimate, lower, upper, sampled row ids)."""
    N = len(rows["proxy"])
    order = np.argsort(rows["proxy"], kind="stable")
    strata = np.array_split(order, K)
    stat = np.ones(N) if agg == "count" else rows["value"]
    pred = rows["pred"]
    n1 = max(int(n * C / K), 2)
    n2 = int(n * (1 - C))
    # Stage 1.
    picks = [rng.choice(s, min(n1, len(s)), replace=False) for s in strata]
    p = np.array([pred[pk].mean() for pk in picks])
    sd = np.array([stat[pk][pred[pk]].std(ddof=1) if pred[pk].sum() > 1 else 0.0 for pk in picks])
    w = np.sqrt(p) * sd
    w = w / w.sum() if w.sum() > 0 else np.full(K, 1 / K)
    # Stage 2 (sample reuse, as in ABAE's default).
    for k in range(K):
        rest = np.setdiff1d(strata[k], picks[k], assume_unique=True)
        extra = min(int(n2 * w[k]), len(rest))
        if extra:
            picks[k] = np.concatenate([picks[k], rng.choice(rest, extra, replace=False)])
    # Stratified estimators (strata have equal size N_k ~ N/K); finite-population correction.
    sizes = np.array([len(s) for s in strata], float)
    tot_mean, tot_var, cnt_mean, cnt_var = 0.0, 0.0, 0.0, 0.0
    cov = 0.0
    for k in range(K):
        pk, Nk = picks[k], sizes[k]
        y = stat[pk] * pred[pk]  # contributes to sum/count
        c = pred[pk].astype(float)
        nk = len(pk)
        fpc = max(1 - nk / Nk, 0.0)
        tot_mean += Nk * y.mean()
        cnt_mean += Nk * c.mean()
        if nk > 1:
            tot_var += Nk**2 * fpc * y.var(ddof=1) / nk
            cnt_var += Nk**2 * fpc * c.var(ddof=1) / nk
            cov += Nk**2 * fpc * np.cov(y, c, ddof=1)[0, 1] / nk
    if agg in ("count", "sum"):
        est, var = tot_mean, tot_var
    else:  # avg = ratio estimator sum / count (delta method)
        if cnt_mean <= 0:
            return math.nan, -math.inf, math.inf, np.concatenate(picks)
        est = tot_mean / cnt_mean
        var = (tot_var - 2 * est * cov + est**2 * cnt_var) / cnt_mean**2
    half = Z * math.sqrt(max(var, 0.0))
    return est, max(est - half, 0.0), est + half, np.concatenate(picks)


def compute_cost(rows, sampled):
    """Summed latency of distinct items evaluated for the sampled rows (all predicates)."""
    total = 0.0
    for _, lat, idx in rows["preds"]:
        total += lat[np.unique(idx[sampled])].sum()
    return total


def run_case(case, agg, target, rng, start_rows=500, growth=1.5):
    """Grow the sample from start_rows by `growth` until the error bound is met (smallest sufficient n)."""
    rows = build_rows(case)
    N = len(rows["proxy"])
    frac = min(start_rows / N, 1.0)
    while True:
        n = min(int(N * frac), N)
        est, lo, hi, sampled = abae_estimate(rows, agg, n, rng)
        err = (hi - lo) / (hi + lo) if hi + lo > 0 else 1.0
        if err <= target or n >= N:
            return {"case": case, "agg": agg, "target": target, "n": n, "N": N, "estimate": est,
                    "lower": lo, "upper": hi, "error": err, "failed": err > target,
                    "time": compute_cost(rows, sampled), "nr_preds": len(rows["preds"])}
        frac = min(frac * growth, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["craigslist_w", "craigslist_b", "craigslist_and", "craigslist_or", "netflix_p",
                             "youtube_a", "youtube_d", "youtube_and", "youtube_or"])
    ap.add_argument("--targets", nargs="+", type=float, default=[0.05, 0.1, 0.5])
    ap.add_argument("--out", default="abae")
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    out = ROOT / "logs" / f"{args.out}.csv"
    for case in args.cases:
        for agg in ["count", "sum", "avg"]:
            for target in args.targets:
                r = run_case(case, agg, target, rng)
                print(r, flush=True)
                pd.DataFrame([r]).to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    sys.exit(main())
