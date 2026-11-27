#!/bin/bash
# Score all benchmark predicates with Gemma, one task at a time (they share the GPU).
# Run from repo root:  caffeinate -dims bash scoring/run_queue.sh > logs/scoring.log 2>&1
set -u
PY=.venv/bin/python

run() { echo "=== START $1 $(date)"; $PY scoring/score_all.py "$1" || echo "=== FAILED $1"; echo "=== END $1 $(date)"; }

# 1) Main benchmark predicates that need no audio.
run netflix_review_positive
run craigslist_title_wood
run craigslist_img_blue_chair

# 2) YouTube: wait for the AudioCaps download, extract clips, then score.
#    (youtube_table() depends on which clips exist, so all clips must be extracted first.)
until grep -q "^EXIT" logs/audiocaps_dl.log 2>/dev/null; do sleep 60; done
echo "=== EXTRACT AUDIO $(date)"
$PY scoring/prepare_audio.py data/audiocaps_parquet/data data/youtube/audio
until [ -f models/gemma-3n-E4B-it-4bit/model-00001-of-00002.safetensors ]; do sleep 30; done
run youtube_description_cooking
run youtube_audio_voices

# 3) Fig. 11 selectivity study: extra image predicates.
for p in leather_sofa wooden_chair dining_table sofa table wooden; do run "craigslist_img_$p"; done
echo "=== ALL DONE $(date)"
