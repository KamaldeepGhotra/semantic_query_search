#!/bin/bash
# YouTube experiments: main benchmark + ablations, Fig. 8 grid, ABAE. Starts once both score caches exist.
cd "$(dirname "$0")"
PY=../.venv-tdb/bin/python
until grep -q "=== ALL DONE" ../logs/scoring_y.log 2>/dev/null; do sleep 30; done
if grep -q "=== FAILED" ../logs/scoring_y.log; then echo "### SCORING FAILED"; exit 1; fi
echo "### YOUTUBE MAIN $(date)"
$PY run_benchmark.py --dbs youtubeaudios --methods local random naive ordered --out main_youtubeaudios
echo "### YOUTUBE FIG8 $(date)"
$PY run_benchmark.py --dbs youtubeaudios --methods local --fig8 --out fig8_youtubeaudios
$PY abae_baseline.py --cases youtube_a youtube_d youtube_and youtube_or --out abae_youtubeaudios
echo "### DONE $(date)"
