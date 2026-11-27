"""Score all Craigslist image predicates in one pass, sharing the image prefix KV cache.

Prompt = [image][question prefix] + ["<predicate>" + answer suffix]. The image and question
prefix are encoded once per image; each predicate suffix runs on a copy-free trimmed cache.
This is mathematically the same forward pass as scoring each predicate separately.

Latency charged per (image, predicate) = prefix time + that predicate's suffix time, i.e. the
cost of an independent single-predicate call (ThalamusDB does not share work across queries).

Usage: python scoring/score_images_multi.py [--limit N] [--check]
Writes data/scores/craigslist_img_<pred>.parquet for every predicate (resumable).
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import pandas as pd
from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm import load
from mlx_vlm.utils import prepare_inputs

sys.path.insert(0, str(Path(__file__).parent))
import items  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = str(ROOT / "models/gemma-3-4b-it-bf16")
PREDS = ["blue chair", "leather sofa", "wooden chair", "dining table", "sofa", "table", "wooden"]
PREFIX = "<bos><start_of_turn>user\n<start_of_image>Does the following text describe the image? "
SUFFIX = '"{pred}"\nAnswer Yes or No.<end_of_turn>\n<start_of_turn>model\n'
CHECKPOINT_EVERY = 200


class MultiScorer:
    def __init__(self):
        self.model, self.processor = load(MODEL)
        tok = self.processor.tokenizer
        self.tok = tok
        self.yes = tok.encode("Yes", add_special_tokens=False)[0]
        self.no = tok.encode("No", add_special_tokens=False)[0]
        self.suffix_ids = {p: mx.array([tok.encode(SUFFIX.format(pred=p), add_special_tokens=False)]) for p in PREDS}

    def _gap(self, logits):
        last = logits[0, -1].astype(mx.float32)
        return (last[self.yes] - last[self.no]).item()

    def score_all(self, img_path):
        """Return {pred: (score, latency)} for one image."""
        t0 = time.perf_counter()
        inp = prepare_inputs(self.processor, prompts=[PREFIX], images=[img_path], image_token_index=262144)
        emb = self.model.get_input_embeddings(inp["input_ids"], inp["pixel_values"], mask=inp["attention_mask"])
        cache = make_prompt_cache(self.model.language_model)
        out = self.model.language_model(inp["input_ids"], inputs_embeds=emb.inputs_embeds, cache=cache)
        mx.eval(out.logits if hasattr(out, "logits") else out, [c.state for c in cache])
        t_prefix = time.perf_counter() - t0
        res = {}
        for p in PREDS:
            t1 = time.perf_counter()
            ids = self.suffix_ids[p]
            out = self.model.language_model(ids, cache=cache)
            gap = self._gap(out.logits if hasattr(out, "logits") else out)
            for c in cache:
                c.trim(ids.shape[1])
            res[p] = (gap, t_prefix + time.perf_counter() - t1)
        return res

    def score_full(self, img_path, pred):
        """Reference: same prompt, no cache sharing (for --check)."""
        inp = prepare_inputs(self.processor, prompts=[PREFIX + SUFFIX.format(pred=pred)], images=[img_path],
                             image_token_index=262144)
        emb = self.model.get_input_embeddings(inp["input_ids"], inp["pixel_values"], mask=inp["attention_mask"])
        out = self.model.language_model(inp["input_ids"], inputs_embeds=emb.inputs_embeds)
        return self._gap(out.logits if hasattr(out, "logits") else out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    paths = items.craigslist_images()
    s = MultiScorer()
    if args.check:
        for path in paths[:4]:
            shared = s.score_all(path)
            for p in PREDS:
                print(f"{Path(path).name:12s} {p:13s} shared={shared[p][0]:7.2f} full={s.score_full(path, p):7.2f} lat={shared[p][1]:.2f}s")
        return
    if args.limit:
        paths = paths[: args.limit]
    outs = {p: ROOT / "data/scores" / f"craigslist_img_{p.replace(' ', '_')}.parquet" for p in PREDS}
    rows = {p: (pd.read_parquet(o).to_dict("records") if o.exists() else []) for p, o in outs.items()}
    start = min(len(r) for r in rows.values())
    rows = {p: r[:start] for p, r in rows.items()}
    print(f"{len(paths)} images, resuming at {start}", flush=True)
    s.score_all(paths[0])  # warm-up
    t0 = time.time()
    for idx in range(start, len(paths)):
        for p, (score, lat) in s.score_all(paths[idx]).items():
            rows[p].append({"idx": idx, "score": score, "latency": lat})
        if (idx + 1) % CHECKPOINT_EVERY == 0 or idx + 1 == len(paths):
            for p in PREDS:
                pd.DataFrame(rows[p]).to_parquet(outs[p])
            rate = (idx + 1 - start) / (time.time() - t0)
            print(f"{idx + 1}/{len(paths)}  {rate:.2f} images/s  eta {(len(paths) - idx - 1) / rate / 60:.1f} min", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
