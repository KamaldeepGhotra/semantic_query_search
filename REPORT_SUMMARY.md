# Reproduction summary: ThalamusDB

COMPSCI 4411/9538 final project, group 1. Category: research exploration in data management systems. Topic:
semantic query processing with LLMs and multimodal models.

This summary covers the reproduction part of the project. Full numbers are in
[`analysis/results.md`](analysis/results.md), and the steps to rerun everything are in [`README.md`](README.md).

## 1. Paper

*ThalamusDB: Approximate Query Processing on Multi-Modal Data* (Saehan Jo and Immanuel Trummer, SIGMOD 2024).

ThalamusDB answers SQL queries that contain natural-language predicates, such as
`nl(img, 'blue chair')` or `nl(audio, 'voices')`, on images, audio and text.

- **Scores and labels.** It does not run an expensive model on every row. Instead, it scores items with zero-shot
  models and asks the user for a few labels, which it uses to calibrate a score threshold.
- **Bounds.** It returns deterministic lower and upper bounds on the query result, computed by rewriting the SQL
  query.
- **Optimizer.** A cost-based optimizer decides, step by step, whether to process more items, ask for another
  label, or stop. It weighs three metrics: result error, number of labels and runtime. The user constrains one of
  them and gives a weight that trades off the other two.

The paper's main claims:

| Claim | Paper value |
|---|---|
| Speedup over exact processing (MindsDB) | 35.0× average, 1,648× max |
| Cost overhead without the optimizer and approximation (ablations: Exact / +Ordered / +Approx) | 951.5× / 945.5× / up to 162× |
| ThalamusDB cheaper than the sampling-based ABAE | 78.9% of cases |
| Bound error under an error constraint | ≤ 10% for every query |
| Post-processing (Table 5) | improves the error in image and audio benchmarks; negligible on Netflix |
| Model inference share of runtime (Fig. 10) | 92.4% (YouTube), 89.6% (Craigslist), 50.7% (Netflix) |

## 2. What we reproduced

All three benchmarks from the paper:

| Benchmark | Data | Queries | NL predicates |
|---|---|---|---|
| YouTube | 46,040 videos with AudioCaps audio clips | 24 | audio "voices", description "cooking" |
| Craigslist | 14,674 furniture images joined with 3,000 listings | 25 | image "blue chair", title "wood" |
| Netflix | 17,770 movies with IMDb reviews, joined with 100M ratings | 6 | review "positive" |

We ran:

- All six constraint settings:
  - Cl, cL: error ≤ 10%, weight 1 or 10⁴.
  - Ce, cE: ≤ 5 labels per predicate.
  - Le, lE: runtime ≤ 10 s.
- All four ablation methods: ThalamusDB, Exact, +Ordered and +Approx.
- The Fig. 8 ABAE grid (error 5/10/50% × weights 1 to 10⁴).
- The Fig. 11 selectivity study (7 image predicates).
- Every metric in Figs. 7–14 and Table 5.

## 3. Method and deviations from the paper

We used the authors' own ThalamusDB code (v1, MIT licence) with as few changes as possible. Where we deviated, it
was forced by hardware, by missing artifacts, or by bugs in that code:

1. **Models.** We could not use the paper's models: CLIP ResNet50, Sentence-BERT and the Mei et al. audio
   retrieval model.
   - We used Gemma 3 4B for text and images, and Gemma 3n E4B for audio, run locally with MLX on an M5 Pro
     MacBook.
   - Score = logit("Yes") − logit("No") for the question "Does the text describe this item?".
   - ThalamusDB only needs a monotone score plus a threshold, so the algorithm is unchanged.
2. **Simulated user labels.** An item satisfies a predicate if and only if its Gemma score is at least 0. The paper
   used thresholds set by hand on its own models' scores.
3. **Runtime.** Each item was scored once, ahead of time, and its measured latency stored. A virtual clock charges
   that latency whenever ThalamusDB evaluates the item, so runtime constraints and optimizer decisions behave as if
   inference ran live.
4. **MindsDB baseline.** We used the authors' Exact baseline instead of MindsDB. Like MindsDB given the true
   thresholds, it evaluates every item.
5. **ABAE.** The authors' ABAE adapter was not released, so we reimplemented ABAE's two-stage stratified sampling
   with the paper's settings:
   - Proxy scores are the oracle scores, at no cost.
   - Labels are free; ABAEmin is charged 3 labels per predicate and ABAEmax is charged the number of labels Exact
     needs.
   - Confidence is 1 − 10⁻⁹, using a normal-approximation interval.
6. **YouTube rows.** We have 46,040 rows against the paper's 46,774 (98.4%); the remaining videos have no audio in
   the AudioCaps release.
7. **Bug fixes in the authors' code.**
   - `run_baseline_ordered` crashed on single-predicate joins.
   - The optimizer raised `UnboundLocalError` when preprocessing alone exceeded the runtime budget.

## 4. Results: paper vs. ours

| Claim | Paper | Ours | Reproduced? |
|---|---|---|---|
| Speedup over exact processing, text queries (Fig. 7) | 35.0× avg | 65.0× avg (Y 61.0×, Cr 63.8×, N 70.0×) | Yes |
| Max speedup | 1,648× | 145× | Lower (see 5.1) |
| Bound error ≤ 10% under Cl and cL | all queries | max 0.100 on all three | Yes |
| Bound error lowest for Min/Max/Limit, highest for Sum/Avg/Count (Fig. 13) | yes | Min/Max/Limit 0–0.02; Sum/Avg/Count 0.05–0.10 | Yes |
| cE beats Ce on relative error, median (Fig. 12) | Y 5.2%, Cr 3.9% vs Ce 100% | Y 18.6%, Cr 4.3% vs Ce 47%, 38% | Yes (same direction) |
| Netflix cE, 80th-percentile relative error | 0.8% vs Ce 44.4% | 0.2% vs Ce 2,028% | Yes |
| F1 of LIMIT queries highest under Cl/cL, lowest under Ce/Le (Fig. 14) | yes | Y 1.00 vs 0.87; Cr 1.00 vs 0.55; N 1.00 everywhere | Yes |
| Post-processing share / error gain, Cl (Table 5) | Y 0.183/0.153, Cr 0.145/0.254, N 0.007/0.001 | Y 0.214/0.105, Cr 0.164/0.198, N 0.037/0.004 | Yes, close |
| Model inference dominates runtime (Fig. 10) | 92.4% / 89.6% / 50.7% | 99.9% / 99.8% / 99.1% | Yes for Y and Cr; not for N (see 5.3) |
| Exact processing overhead (Fig. 9) | 951.5× | 66.7× | Same direction, smaller |
| +Ordered overhead | 945.5× | 58.2× | Same direction, smaller |
| +Approx overhead | Y 34.0×, Cr 162.1×, N 6.0× | Y 3.6×, Cr 3.7×, N 3.1× | Same direction, smaller |
| Big gains for Min/Max/Limit, small for Count/Avg (Fig. 11) | yes | 50–97× vs 1.0–1.1× | Yes |
| Larger gains for less selective predicates (Fig. 11) | tendency | weak (Sum rises from 1.1× to 2.1×) | Partly |
| ThalamusDB cheaper than ABAE (Fig. 8) | 78.9% overall (ABAEmax 91.1%, ABAEmin 67.7%) | 32.7% overall (ABAEmax 49.9%, ABAEmin 15.6%) | No (see 5.2) |
| ABAE fails to meet the error bound in some cases | yes (dark cells in Fig. 8) | 9.9% of (query, bound) pairs | Yes |

## 5. Discussion of the differences

**5.1 Smaller speedups and ablation overheads.** The paper's largest numbers come from cases where exact processing
with its models was very expensive. On our hardware Gemma scores an item in 0.05–0.9 s, and the tables are bounded
(at most 46,040 items), so the cost of the Exact baseline is capped much lower. Every ablation still ranks the same
way as in the paper: ThalamusDB < +Approx < +Ordered ≤ Exact.

**5.2 ABAE does better than in the paper.** The paper gives ABAE oracle-quality proxy scores at no cost. In our
setup the proxy and the simulated ground truth are both Gemma scores, so ABAE's strata are perfectly sorted, and a
small sample meets the bound. The paper's proxies came from models other than the ones defining ground truth, so
they were noisier. On YouTube, where the audio predicate is less clean, ABAE failed to meet the bound in about 10%
of cases, which matches the paper's observation.

**5.3 Netflix runtime breakdown.** The paper reports 45.5% relational time on Netflix, because of its 100M-row
ratings join. With DuckDB 2.0 on an M5 Pro that join is fast (0.8% of runtime), so inference dominates (99.1%).

**5.4 The 10-second runtime constraints (Le, lE).** Gemma's preprocessing (scoring the first sample and requesting
the first labels) already takes about 25 s, more than the paper's 10 s budget. Le and lE therefore stop with wide
bounds. We kept the paper's 10 s budget rather than rescaling it, to stay faithful.

**5.5 Audio model quality.** Gemma 3n is strong on speech but weak on environmental sound. On 100 AudioCaps clips
it separates "voices" from non-voice clips with an AUC of 0.73, and it labels 88.7% of clips as containing voices.
Because the simulated labels come from the same model, this does not affect what ThalamusDB is being tested on.
It does make the YouTube audio predicate much less selective than in the paper.

## 6. Conclusion

Most of ThalamusDB's qualitative claims hold with a completely different model family (Gemma instead of
CLIP/BERT/audio retrieval):

- Large speedups over exact processing at ≤ 10% error.
- Deterministic bounds that always meet the error constraint.
- The expected ordering of the ablations.
- High F1 on LIMIT queries.
- Post-processing that helps on image and audio data but not on Netflix.

The absolute sizes of the speedups and ablation overheads are much smaller on our hardware, and ThalamusDB's
advantage over ABAE does not reproduce. Both differences trace back to setup choices that the paper's evaluation
depends on: how expensive exact processing is, and how good ABAE's free proxy is.

## 7. Still to do for the project (not part of this reproduction)

The project brief also asks for an idea extension, slides, a recorded 15-minute presentation and the final report.
Candidate extensions that follow from these results:

- Rerun Le/lE with the runtime budget scaled to Gemma's preprocessing time.
- Give ABAE a cheap, separate proxy model (for example a small CLIP) so its proxy is no longer perfect.
- Use an LLM's Yes/No probabilities directly as calibrated scores, or compare ThalamusDB v2, the authors' LLM-based
  version.
- Measure accuracy against human labels (for example IMDb review sentiment, which Gemma matches 90.7% of the time).

## 8. Artifacts

| Path | Contents |
|---|---|
| `scoring/` | Gemma scorer and scoring scripts. |
| `tdb/` | Adapted ThalamusDB code, benchmark driver and ABAE baseline. |
| `analysis/analyze.py`, `analysis/results.md` | Every figure and table metric. |
| `data/scores/` | Gemma score caches, so the experiments can be rerun without a GPU. |
| `logs/*.csv` | Raw per-run results. |

Dataset sources (the authors' Google Drive links, and AudioCaps via Hugging Face) are listed in `README.md`.
