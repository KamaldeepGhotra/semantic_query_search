# Semantic Query Search: reproducing ThalamusDB

COMPSCI 4411/9538 final project, group 1. Topic: semantic query processing with LLMs and multimodal models.

This repository reproduces the experiments of *ThalamusDB: Approximate Query Processing on Multi-Modal Data*
(Jo and Trummer, SIGMOD 2024) on all three of the paper's benchmarks: Craigslist (images and text), YouTube
(audio and text), and Netflix (text). The zero-shot models from the paper (CLIP, Sentence-BERT, and an audio
retrieval model) are replaced by Gemma models that run locally on Apple Silicon. Everything else follows the
paper: the same queries, the same six constraint settings, the same baselines, and the authors' own ThalamusDB code.

The paper-vs-ours comparison is in [`analysis/results.md`](analysis/results.md), and a write-up for the final
report is in [`REPORT_SUMMARY.md`](REPORT_SUMMARY.md).

## Layout

| Path | Contents |
|---|---|
| `scoring/` | Gemma zero-shot scorer and scripts that score every item of every benchmark predicate once. |
| `tdb/` | The authors' ThalamusDB v1 code (MIT, see `tdb/LICENSE-thalamusdb`), adapted to read cached Gemma scores, plus the benchmark driver and the ABAE baseline. |
| `analysis/` | `analyze.py` turns the result CSVs into every figure and table claim; the output is `results.md`. |
| `data/scores/` | Gemma score caches (`idx`, `score`, `latency` per item), so the experiments can be rerun without a GPU. |
| `logs/*.csv` | Raw per-run results (one row per query, configuration and method). |

## How it works

**Scores.** For an item (text, image or audio clip) and a predicate, the score is
`logit("Yes") - logit("No")` from Gemma's answer to *"Does the following text describe the image/audio clip/match
the description …? Answer Yes or No."* ThalamusDB only needs a monotone similarity score and a threshold, so this
plugs into its threshold-calibration design unchanged.

**Models.**

- Gemma 3 4B (`mlx-community/gemma-3-4b-it-bf16`) for text and images.
- Gemma 3n E4B (`mlx-community/gemma-3n-E4B-it-4bit`) for audio.
- Both run with `mlx-vlm` on an M5 Pro MacBook with 24 GB of RAM.

**Simulated user labels.** An item satisfies a predicate if and only if its Gemma score is at least 0, which is
the model's own Yes/No decision boundary. The paper used per-predicate thresholds on CLIP/BERT scores that the
authors set by hand.

**Runtime.** Every item is scored once, ahead of time, and the measured latency is stored. During the experiments,
a virtual clock (`tdb/simclock.py`) adds the cached latency of each item ThalamusDB chooses to evaluate. As a
result, runtimes, runtime constraints and optimizer decisions behave as if Gemma were running live. SQL and
optimizer time is measured for real.

**Methods.** These use the authors' names.

| Name | What it is |
|---|---|
| `local` | ThalamusDB. |
| `naive` | Exact processing; this stands in for MindsDB. |
| `ordered` | +Ordered. |
| `random` | +Approx. |
| ABAE | Reimplemented in `tdb/abae_baseline.py`, because the authors' ABAE adapter was not released. |

## Reproducing

### 1. Environments (Python 3.14)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-scoring.txt
python3 -m venv .venv-tdb && .venv-tdb/bin/pip install -r requirements-tdb.txt
```

### 2. Datasets (git-ignored, about 50 GB)

These are the same sources the authors use.

```bash
# Craigslist images and tables (authors' Google Drive link from the ThalamusDB README), into data/craigslist/
gdown 1iy_H4jjeDDnDHY1wi1l6A6THodB2ShFY
# Netflix ratings, 2.7 GB, into data/netflix/
gdown 11lfRbEtJbdot_G3Qlr3lb-85ZYTrbC3c
# AudioCaps clips for YouTube (Hugging Face mirror of the AudioCaps waveforms), then convert to 16 kHz FLAC
hf download OpenSound/AudioCaps --repo-type dataset --local-dir data/audiocaps_parquet
.venv/bin/python scoring/prepare_audio.py data/audiocaps_parquet/data data/youtube/audio
```

The CSV files (`furnitures.csv`, `imgs.csv`, `movies_with_reviews.csv`, `youtube.csv`) come from the authors'
repository (`https://github.com/saehanjo/thalamusdb`).

### 3. Models

```bash
hf download mlx-community/gemma-3-4b-it-bf16 --local-dir models/gemma-3-4b-it-bf16
hf download mlx-community/gemma-3n-E4B-it-4bit --local-dir models/gemma-3n-E4B-it-4bit
```

### 4. Scoring

Skip this step if you only want to rerun the experiments: the caches are already in `data/scores/`.

```bash
bash scoring/run_queue_cn.sh      # Netflix reviews, Craigslist titles, 7 Craigslist image predicates
bash scoring/run_queue_y.sh       # YouTube descriptions and audio
```

### 5. Experiments and analysis

```bash
bash tdb/run_cn_experiments.sh    # Craigslist and Netflix: main benchmark, ablations, Fig. 8 grid, ABAE
bash tdb/run_fig11.sh             # Fig. 11 selectivity study (Craigslist)
bash tdb/run_y_experiments.sh     # YouTube: main benchmark, ablations, Fig. 8 grid, ABAE
.venv-tdb/bin/python analysis/analyze.py   # writes analysis/results.md
```

## Changes to the authors' code

The changes in `schema.py` are marked with `added` comments.

- `nlfilter.py`: the CLIP/BERT/audio processors are replaced by `CachedProcessor`, which reads Gemma score caches
  and advances the virtual clock.
- `nldbs.py`: a new data loader for the local file layout. Unused `neighborhood_u` and `url_u` columns are dropped.
- `schema.py`: extra result fields are logged (bounds, LIMIT rows for F1, post-processing counts), and a guard is
  added in `run_baseline_ordered` for single-predicate joins, which raised `IndexError`.
- `optimizer.py`: `cur_cost` and `cur_error` are now assigned before the runtime-constraint `break`. Without this,
  `UnboundLocalError` is raised when preprocessing alone exceeds the runtime budget.
- The `streamlit` import is removed from `schema.py` and `nlfilter.py`.
