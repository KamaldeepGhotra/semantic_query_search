#!/bin/bash
# Netflix main benchmark, configs after the optimizer crash fix (Cl, cL already done).
cd "$(dirname "$0")"
until [ -f ../logs/abae_netflix.csv ] && [ "$(wc -l < ../logs/abae_netflix.csv)" -ge 10 ]; do sleep 30; done
echo "### NETFLIX REST $(date)"
../.venv-tdb/bin/python run_benchmark.py --dbs netflix --methods local random --configs Le lE Ce cE --out main_netflix
echo "### NETFLIX REST DONE $(date)"
