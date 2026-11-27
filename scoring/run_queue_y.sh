#!/bin/bash
# YouTube scoring: descriptions (Gemma 3 4B) then audio (Gemma 3n E4B). Run from repo root. Resumable.
PY=.venv/bin/python
run() { echo "=== START $1 $(date)"; $PY scoring/score_all.py "$1" || echo "=== FAILED $1"; echo "=== END $1 $(date)"; }
run youtube_description_cooking
run youtube_audio_voices
echo "=== ALL DONE $(date)"
