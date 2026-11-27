"""Score every item of one benchmark column against one NL predicate with Gemma; cache to parquet.

Usage: python scoring/score_all.py <task> [--limit N]
Tasks are defined in TASKS below. Output: data/scores/<task>.parquet with columns
idx (row position in the ThalamusDB column), score, latency (seconds).
Resumable: re-running continues from the last checkpoint.
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import items  # noqa: E402
from gemma_scorer import GemmaScorer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GEMMA3 = str(ROOT / "models/gemma-3-4b-it-bf16")
GEMMA3N = str(ROOT / "models/gemma-3n-E4B-it-4bit")

# task -> (item loader, modality, model, predicate). Predicates are those in the authors' benchmark.py.
TASKS = {
    "craigslist_img_blue_chair": (items.craigslist_images, "image", GEMMA3, "blue chair"),
    "craigslist_title_wood": (items.craigslist_titles, "text", GEMMA3, "wood"),
    "youtube_description_cooking": (items.youtube_descriptions, "text", GEMMA3, "cooking"),
    "youtube_audio_voices": (items.youtube_audio, "audio", GEMMA3N, "voices"),
    "netflix_review_positive": (items.netflix_reviews, "text", GEMMA3, "positive"),
}
# Fig. 11 selectivity study: extra image predicates on the same column.
for _p in ["leather sofa", "wooden chair", "dining table", "sofa", "table", "wooden"]:
    TASKS[f"craigslist_img_{_p.replace(' ', '_')}"] = (items.craigslist_images, "image", GEMMA3, _p)

CHECKPOINT_EVERY = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    loader, modality, model_path, pred = TASKS[args.task]
    data = loader()
    if args.limit:
        data = data[: args.limit]
    out = ROOT / "data/scores" / f"{args.task}{'_sample' if args.limit else ''}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(out).to_dict("records") if out.exists() else []
    start_idx = len(rows)
    print(f"{args.task}: {len(data)} items, resuming at {start_idx}", flush=True)

    # Cap MLX's buffer cache: varying input lengths otherwise grow it far past the model size.
    mx.set_cache_limit(512 * 1024**2)
    scorer = GemmaScorer(model_path, modality)
    scorer.score(pred, data[0])  # Warm-up so compile time is not charged to item 0.
    t0 = time.time()
    for idx in range(start_idx, len(data)):
        score, latency = scorer.score(pred, data[idx])
        rows.append({"idx": idx, "score": score, "latency": latency})
        if (idx + 1) % CHECKPOINT_EVERY == 0 or idx + 1 == len(data):
            pd.DataFrame(rows).to_parquet(out)
            done = idx + 1 - start_idx
            rate = done / (time.time() - t0)
            print(f"{idx + 1}/{len(data)}  {rate:.2f} items/s  eta {(len(data) - idx - 1) / rate / 60:.1f} min", flush=True)
    pd.DataFrame(rows).to_parquet(out)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
