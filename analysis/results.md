# ThalamusDB reproduction: paper vs ours
Models: Gemma 3 4B (text, images) and Gemma 3n E4B (audio) replace Sentence-BERT/CLIP/the audio retrieval model. Simulated labels use Gemma's Yes/No boundary (score >= 0). Runtime = measured Gemma latency (M5 Pro) + real SQL/optimizer time.

## Fig. 7 — speedup over exact processing (text-predicate queries, error <= 10%)
Paper compares to MindsDB (exact, given true thresholds); we use the Exact baseline, which is the same work.
| | Paper | Ours |
|---|---|---|
| Queries | 12 (incl. YouTube) | 18 |
| Average speedup | 35.0x | 65.0x |
| Max speedup | 1,648.4x | 145.1x |
- craigslist: mean 63.8x over 6 queries
- netflix: mean 70.0x over 6 queries
- youtubeaudios: mean 61.0x over 6 queries

## Fig. 9 — ablations, cost relative to ThalamusDB (=1)
| Benchmark | Config | Exact | +Ordered | +Approx |
|---|---|---|---|---|
| youtubeaudios | Cl | 170.3 | 144.6 | 1.5 |
| youtubeaudios | cL | 3.0 | 2.8 | 8.8 |
| youtubeaudios | Ce | 434.8 | 360.1 | 7.8 |
| youtubeaudios | cE | 172.2 | 152.4 | 1.3 |
| youtubeaudios | Le | 3.4 | 3.2 | 1.0 |
| youtubeaudios | lE | 1.3 | 1.3 | 1.0 |
| craigslist | Cl | 27.1 | 21.1 | 2.3 |
| craigslist | cL | 2.0 | 1.9 | 1.8 |
| craigslist | Ce | 97.8 | 74.5 | 11.2 |
| craigslist | cE | 24.7 | 21.6 | 5.1 |
| craigslist | Le | 2.3 | 2.1 | 1.0 |
| craigslist | lE | 0.5 | 0.5 | 1.0 |
| netflix | Cl | 63.1 | 63.1 | 1.2 |
| netflix | cL | 2.7 | 2.7 | 2.1 |
| netflix | Ce | 126.4 | 126.4 | 11.3 |
| netflix | cE | 65.0 | 65.0 | 1.9 |
| netflix | Le | 2.3 | 2.3 | 1.0 |
| netflix | lE | 1.3 | 1.3 | 1.0 |

| Average overhead | Paper | Ours |
|---|---|---|
| Exact (all benchmarks) | 951.5x | 66.7x |
| +Ordered (all benchmarks) | 945.5x | 58.2x |
| +Approx (craigslist) | 162.1x | 3.7x |
| +Approx (netflix) | 6.0x | 3.1x |
| +Approx (youtubeaudios) | 34.0x | 3.6x |

## Fig. 10 — runtime breakdown of ThalamusDB (average over configs and queries)
| Benchmark | Model inference (paper) | Model inference (ours) | Relational (paper) | Relational (ours) | Optimization (ours) |
|---|---|---|---|---|---|
| youtubeaudios | 92.4% | 99.9% | 4.1% | 0.1% | 0.0% |
| craigslist | 89.6% | 99.8% | 2.5% | 0.1% | 0.0% |
| netflix | 50.7% | 99.1% | 45.5% | 0.8% | 0.0% |

## Fig. 12 / 13 — bound error and relative error
Relative error = |true - midpoint(bounds)| / true; true value from the Exact baseline.

| Benchmark | Config | max bound error | median bound error | median relative error | 80th pct relative error |
|---|---|---|---|---|---|
| youtubeaudios | Cl | 0.100 | 0.016 | 0.046 | 0.075 |
| youtubeaudios | cL | 0.100 | 0.027 | 0.047 | 0.069 |
| youtubeaudios | Ce | 1.000 | 0.820 | 0.469 | 83.244 |
| youtubeaudios | cE | 1.000 | 0.144 | 0.186 | 3.034 |
| youtubeaudios | Le | 1.000 | 0.844 | 0.469 | 89.616 |
| youtubeaudios | lE | 1.000 | 0.844 | 0.469 | 97.146 |
| craigslist | Cl | 0.100 | 0.000 | 0.000 | 0.072 |
| craigslist | cL | 0.099 | 0.053 | 0.017 | 0.066 |
| craigslist | Ce | 1.000 | 0.993 | 0.381 | 106.170 |
| craigslist | cE | 1.000 | 0.025 | 0.043 | 6.369 |
| craigslist | Le | 1.000 | 0.990 | 0.715 | 74.121 |
| craigslist | lE | 1.000 | 0.990 | 0.715 | 74.121 |
| netflix | Cl | 0.099 | 0.047 | 0.004 | 0.007 |
| netflix | cL | 0.099 | 0.047 | 0.004 | 0.008 |
| netflix | Ce | 1.000 | 0.494 | 0.016 | 20.281 |
| netflix | cE | 0.062 | 0.003 | 0.000 | 0.002 |
| netflix | Le | 1.000 | 0.495 | 0.016 | 17.209 |
| netflix | lE | 1.000 | 0.496 | 0.016 | 25.761 |

Paper reference points: cE median relative error 5.2% (YouTube), 3.9% (Craigslist) vs Ce 100%; Netflix cE 80th pct 0.8% vs Ce 44.4%. Bound error <= 10% for every query under Cl and cL.

**Fig. 13 — mean bound error by aggregate type (error <= 10%)**

| Benchmark | Config | Min | Max | Sum | Avg | Cnt | Limit |
|---|---|---|---|---|---|---|---|
| youtubeaudios | Cl | 0.000 | 0.000 | 0.099 | 0.096 | 0.081 | 0.000 |
| youtubeaudios | cL | 0.000 | 0.000 | 0.099 | 0.092 | 0.085 | 0.000 |
| craigslist | Cl | 0.000 | 0.018 | 0.082 | 0.045 | 0.049 | 0.000 |
| craigslist | cL | 0.000 | 0.018 | 0.085 | 0.085 | 0.067 | 0.021 |
| netflix | Cl | 0.000 | 0.000 | 0.099 | 0.098 | 0.094 | 0.000 |
| netflix | cL | 0.000 | 0.000 | 0.099 | 0.095 | 0.097 | 0.000 |

## Fig. 14 — F1 of LIMIT queries

| Benchmark | Cl | cL | Ce | cE | Le | lE |
|---|---|---|---|---|---|---|
| youtubeaudios | 1.00 | 1.00 | 0.87 | 1.00 | 0.87 | 0.87 |
| craigslist | 1.00 | 0.98 | 0.55 | 0.99 | 0.55 | 0.55 |
| netflix | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Paper: highest F1 under Cl/cL, then cE/lE, lowest under Ce/Le.

## Table 5 — post-processing (error <= 10%)

| Benchmark | Config | #post / #all actions (ours) | error improvement (ours) |
|---|---|---|---|
| youtubeaudios | Cl | 0.214 | 0.105 |
| youtubeaudios | cL | 0.218 | 0.103 |
| craigslist | Cl | 0.164 | 0.198 |
| craigslist | cL | 0.208 | 0.153 |
| netflix | Cl | 0.037 | 0.004 |
| netflix | cL | 0.013 | 0.002 |

Paper: YouTube Cl 0.183 / 0.153, cL 0.248 / 0.246; Craigslist Cl 0.145 / 0.254, cL 0.170 / 0.247; Netflix Cl 0.007 / 0.001, cL 0.000 / 0.000.

## Fig. 8 — ABAE vs ThalamusDB (Count/Sum/Avg; error 5/10/50%; weight 1..1e4)
| | Paper | Ours |
|---|---|---|
| ThalamusDB cheaper than ABAEmax | 91.1% | 49.9% |
| Avg cost overhead of ABAEmax (excl. ABAE failures) | 2.4x | 1.5x |
| ThalamusDB cheaper than ABAEmin | 67.7% | 15.6% |
| Avg cost overhead of ABAEmin (excl. ABAE failures) | 2.1x | 1.0x |
| ThalamusDB beats ABAE, all cases (abstract) | 78.9% | 32.7% |
| ABAE failed to meet error bound | — | 9.9% of (query, bound) |

ThalamusDB win rate by error bound: 0.05: 31.1%, 0.1: 33.3%, 0.5: 33.7% (paper: margin grows as the bound tightens).

## Fig. 11 — ThalamusDB vs Exact by predicate selectivity (Craigslist images, error <= 10%, w = 1)

| Predicate | Selectivity | Min | Max | Sum | Avg | Cnt | Limit |
|---|---|---|---|---|---|---|---|
| blue chair | 0.018 | 49.5x | 49.5x | 1.1x | 1.0x | 1.0x | 32.4x |
| leather sofa | 0.043 | 97.0x | 97.0x | 1.4x | 1.0x | 1.0x | 48.2x |
| dining table | 0.123 | 97.4x | 49.6x | 1.4x | 1.0x | 1.0x | 92.8x |
| sofa | 0.140 | 97.3x | 97.1x | 1.8x | 1.0x | 1.0x | 92.5x |
| wooden chair | 0.147 | 97.3x | 97.3x | 1.4x | 1.0x | 1.0x | 92.7x |
| table | 0.376 | 97.0x | 96.9x | 1.9x | 1.1x | 1.1x | 92.4x |
| wooden | 0.523 | 97.2x | 97.2x | 2.1x | 1.1x | 1.1x | 92.6x |

Cells = Exact cost / ThalamusDB cost. Paper: large gains for Min/Max/Limit, smaller for Count/Avg, and a tendency toward larger gains for less selective predicates.
