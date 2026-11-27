#!/bin/bash
# Craigslist + Netflix experiments, started as soon as each score cache is complete.
# Run from repo root:  nohup caffeinate -dims bash tdb/run_cn_experiments.sh > logs/experiments_cn.log 2>&1 &
cd "$(dirname "$0")"
PY=../.venv-tdb/bin/python
wait_for() { until grep -q "=== END $1" ../logs/scoring_cn.log 2>/dev/null; do sleep 30; done; }

# Netflix: main benchmark (Figs. 9, 10, 12-14, Table 5), ablations, Fig. 8 grid, ABAE.
wait_for netflix_review_positive
echo "### NETFLIX MAIN $(date)"
$PY run_benchmark.py --dbs netflix --methods local random naive ordered --out main_netflix
echo "### NETFLIX FIG8 $(date)"
$PY run_benchmark.py --dbs netflix --methods local --fig8 --out fig8_netflix
$PY abae_baseline.py --cases netflix_p --out abae_netflix

# Craigslist: same set.
wait_for craigslist_img_blue_chair
echo "### CRAIGSLIST MAIN $(date)"
$PY run_benchmark.py --dbs craigslist --methods local random naive ordered --out main_craigslist
echo "### CRAIGSLIST FIG8 $(date)"
$PY run_benchmark.py --dbs craigslist --methods local --fig8 --out fig8_craigslist
$PY abae_baseline.py --cases craigslist_w craigslist_b craigslist_and craigslist_or --out abae_craigslist
echo "### DONE $(date)"
