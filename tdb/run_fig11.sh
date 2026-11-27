#!/bin/bash
# Fig. 11 (selectivity): ThalamusDB vs Exact, error <= 10%, weight 1 (Cl). Starts after main runs.
cd "$(dirname "$0")"
until grep -q "### DONE" ../logs/experiments_cn.log 2>/dev/null; do sleep 30; done
echo "### FIG11 $(date)"
../.venv-tdb/bin/python run_benchmark.py --dbs craigslist --methods local naive --configs Cl --fig11 --out fig11_craigslist
echo "### FIG11 DONE $(date)"
