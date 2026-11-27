#!/bin/bash
# Craigslist + Netflix scoring (YouTube deferred). Run from repo root. Resumable.
PY=.venv/bin/python
run() { echo "=== START $1 $(date)"; $PY scoring/score_all.py "$1" || echo "=== FAILED $1"; echo "=== END $1 $(date)"; }
run netflix_review_positive
run craigslist_title_wood
# All 7 image predicates (main benchmark 'blue chair' + Fig. 11 set) share one image pass.
echo "=== START craigslist_img_all $(date)"
$PY scoring/score_images_multi.py || echo "=== FAILED craigslist_img_all"
echo "=== END craigslist_img_blue_chair $(date)"
echo "=== ALL DONE $(date)"
