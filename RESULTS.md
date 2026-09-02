# Every measurement, with its conditions

One NVIDIA DGX Spark (GB10, 128 GB unified, ~273 GB/s spec, ~231 GB/s measured
effective). SGLang, `RadixArk/Qwen3.8-27B-NVFP4`, DFlash2 drafter, draft budget 16,
2026-09-01 and 2026-09-02.

**Counting and methodology, read before comparing.** Generation is decode only:
completion tokens divided by (wall time minus time to first token), so prefill is
never counted as decode. Prefill is prompt tokens divided by the cold time to
first token. Accepted tokens per pass comes from the server's own
`sglang:spec_accept_length` gauge. Prompt sizes are calibrated against the
server's tokenizer, so "8k" is 8,192 prompt tokens. Thinking is off. Each table
states its temperature, its reps and its boots. Most are **one boot**, and one
boot is not a distribution. Raw rows are in `data/`; `bench/plotx.py` rebuilds every
chart from them.

## 1. The recommended recipe across the context ladder

`bench/ctxsweep.py`, NVFP4 drafter, bf16 KV, draft 16, 512 output tokens, **4 reps
per rung, median**, temperature 0, `MAX_RUNNING=4`. Two boots, 2026-09-02: boot 1
at 03:51 (`data/ctxsweep-recommended.csv`, the source of `charts/ctxsweep-27b.png`)
and boot 2 later the same day on the same, verified-intact tactic draw.

| Prompt tok | Prefill tok/s | Gen, boot 1 | Gen, boot 2 | Accepted tok/pass | vs the bf16 drafter |
|---|---|---|---|---|---|
| 324 | 1,657 | **72.7** | 75.6 | 6.88 | +0% |
| 1,099 | 2,310 | **74.8** | 78.7 | 7.47 | +6% |
| 2,307 | 1,862 | **70.7** | 76.2 | 7.20 | +7% |
| 4,311 | 2,474 | **78.3** | 77.5 | 7.78 | +16% |
| 8,159 | 2,506 | **73.9** | 75.5 | 7.53 | +10% |
| 16,576 | 2,382 | **72.7** | 78.2 | 7.28 | +16% |
| 32,741 | 2,181 | **66.8** | 65.8 | 7.17 | −3% |
| 65,842 | 1,811 | **66.8** | 63.8 | 8.12 | −4% |
| 130,656 | 1,372 | **55.9** | 56.4 | 7.92 | +14% |
| 257,334 | 928 | **38.2** | 41.3 | 6.83 | +6% |

The comparison column is the bf16 drafter measured the same way: the 2-rep sweep to
130k, and the 4-rep long tail's 260k rung for the ceiling. Prefill and acceptance
are unchanged by the swap (6.8-8.1 tokens per pass against 6.9-9.0).

**Boot to boot, median absolute difference 4.3%, worst 8.1%, mean +2.9%** in boot
2's favour. Prefill agrees to a few percent past 4k, and the tactic cache was
byte-identical across the two boots, so the gaps are acceptance draws on free-form
output (7.1-8.3 against 6.8-8.1 tokens per pass), not the server. **Quote this
recipe as "about 70 tok/s to 65k across two boots". Do not quote a single rung to
three digits.** Boot 2's KV pool was 413,460 tokens at `MAX_RUNNING=4`.

### Temperature 1 on the same server

Same server, same prompts, `--temperature 1.0`, **4 reps, median**, 2026-09-02
08:05. Data: the `temperature=1.0` rows of the same CSV.

| Prompt tok | Gen tok/s, T=0 | Gen tok/s, T=1 | Accepted tok/pass, T=0 → T=1 |
|---|---|---|---|
| ~320 | 72.7 | 72.7 | 6.9 → 7.7 |
| 8k | 73.9 | 66.6 (−10%) | 7.5 → 6.4 |
| 65k | 66.8 | 60.8 (−9%) | 8.1 → 6.3 |
| 130k | 55.9 | 50.2 (−10%) | 7.9 → 7.8 |
| 260k | 38.2 / 41.3 | 39.7 | 7.2 → 6.8 |

Sampling costs about 10% from 8k to 130k, and it costs it through acceptance: the
verify step matches the draft against a sample, not an argmax. Prefill and TTFT do
not depend on the sampler and do not move. **At the ceiling the penalty vanishes**:
39.7 tok/s at 260k sits inside the two T=0 boots' 38.2-41.3, because that pass is
KV-bound rather than acceptance-bound. That rung needed a `MAX_RUNNING=4` boot; on
the `MAX_RUNNING=16` boot the 238,605-token pool refuses the input with HTTP 400.

## 2. Three recipes, and the prefix cache, at each rung

Same box, same prompts, same tactic draw, temperature 0, 512 output tokens, one
boot each. The bf16 drafter and the candidate ran 2 reps per rung (2026-09-01), the
recommended recipe 4 (2026-09-02). Generation is tokens per second. The TTFT columns
are the bf16-drafter run; the prefix cache does not depend on the drafter. Data:
`data/ctxsweep-three-recipes.csv`. Chart: `charts/three-recipes.png`.

| Rung | bf16 drafter, bf16 KV | NVFP4, bf16 KV | NVFP4, fp8 KV | Cold TTFT | Warm TTFT | Cold ÷ warm |
|---|---|---|---|---|---|---|
| ~320 | 72.7 | 72.7 | **77.7** | 0.19 s | 0.22 s | 1x |
| ~1.1k | 70.8 | 74.8 | 74.6 | 0.52 s | 0.30 s | 2x |
| ~2.3k | 66.0 | 70.7 | **73.2** | 0.95 s | 0.24 s | 4x |
| ~4.2k | 67.2 | **78.3** | 73.3 | 1.82 s | 0.26 s | 7x |
| ~8.2k | 67.1 | 73.9 | **74.9** | 3.36 s | 0.20 s | 17x |
| ~16.5k | 62.6 | 72.7 | **77.8** | 6.97 s | 0.28 s | 25x |
| ~32.7k | 68.8 | 66.8 | **80.7** | 14.93 s | 0.34 s | 44x |
| ~65.6k | 69.7 | 66.8 | 61.5 | 35.91 s | 0.47 s | 77x |
| ~130k | 49.2 | 55.9 | **63.3** | 94.01 s | 0.70 s | 134x |
| ~200k / 260k | 43.3 at 199k | 38.2 at 257k | see section 7 | 182.01 s | 0.94 s | **193x** |

The last row is two ceilings: the bf16 run's calibration landed at 199,008 tokens
and the 4-rep run's at 257,334, from the same 262,144 target. The candidate boot's
own ceiling rung is in the CSV but is not quoted, because section 7 measured that
depth directly. The candidate is up at every rung but 65k, where acceptance drew
low (7.5 against 8.95), and its prefill is **down 3-21%**. See section 7.

Warm TTFT grows from 0.22 s to 0.94 s while the prompt grows 600x; cold TTFT grows
950x, as it must. The gap is the prefix cache, 193x at 199k. Prefill on that run
peaked at 2,456 tok/s (8k) and held 1,093 at 199k, and time per output token is
14-16 ms flat to 66k, rising to 23 ms at 199k.

## 3. The long tail, re-measured

`ctxsweep.py --reps 4`, bf16 drafter, same server, 2026-09-01 20:30. `ms/pass`
is accepted tokens per pass divided by generation: one draft-plus-verify cycle.

| Prompt tok | Prefill tok/s | Gen tok/s | Accepted tok/pass | ms/pass | Cold TTFT | Warm TTFT |
|---|---|---|---|---|---|---|
| 65,507 | 1,846 | 68.5 | 8.83 | 129 | 35.5 s | 0.46 s |
| 97,618 | 1,588 | 52.6 | 6.73 | 128 | 61.5 s | 0.55 s |
| 130,121 | 1,387 | 52.6 | 8.08 | 154 | 93.9 s | 0.68 s |
| 164,890 | 1,223 | 44.0 | 6.98 | 158 | 134.9 s | 0.79 s |
| 260,557 | 929 | 35.9 | 5.98 | 166 | 280.5 s | 1.18 s |

From 65k to 260k generation falls to 0.52x. That splits as acceptance 0.68x
(8.83 → 5.98) times pass rate 0.77x (129 → 166 ms): the drafter losing confidence is
the larger factor, not the KV read. A straight-line fit over all 15 rungs of both
runs gives `ms/pass = 110 + 0.279 per 1k context tokens` (R² 0.86). Worked back
through the target's 65,536 bytes per token that is **235 GB/s**, the box's bus.

## 4. Concurrency

Measured with `bench/concbench.py`; the aggregate and per-stream ladder gets its
own write-up. One finding belongs here because it is a configuration fact:
`MAX_RUNNING=16` leaves a KV pool of 238,605 tokens against 413,460 at
`MAX_RUNNING=4`, so inputs above ~238k are refused on a 16-seat boot.

## 5. Fixed-output acceptance

`bench/accfix.py`: at every prefix depth the model reproduces the same 68-line
class verbatim, so the wanted output is identical and only the drafter's
conditioning can change. 4 reps, temperature 0, exact match on all 16 runs, one
boot, 2026-09-01 21:04. Saturation is accepted tokens over the budget of 16.

| Prompt tok | Gen tok/s | Accepted tok/pass | Saturation | ms/pass |
|---|---|---|---|---|
| 1,166 | 130.5 | 14.45 | 90% | 111 |
| 65,358 | 111.2 | 14.48 | 90% | 130 |
| 131,720 | 99.8 | 14.45 | 90% | 145 |
| 259,633 | 78.6 | 13.53 | 85% | 172 |

Acceptance moves 0.2% from 1k to 130k and 6% at 260k. The 8.8 → 6.0 fall in the
free-form sweep is therefore the model writing a less predictable answer with 260k
of code in front of it, not the drafter losing its conditioning. Because this test
is stable to 0.2% on bf16, it is also the cleanest read on the drafter swap
(2026-09-01, same boot pair):

| Test | bf16 drafter | NVFP4 drafter |
|---|---|---|
| Fixed output, 1k prefix: tok/pass · gen | 14.45 · 130.5 | 14.53 · **143.1** (+9.6%) |
| Fixed output, 65k prefix: tok/pass · gen | 14.48 · 111.2 | 13.95 · **117.2** (+5.4%) |
| Sweep task, 320 tok, 6 reps × 2 boots | 63.2-65.2 · 7.1-7.5 | 59.8-70.4 · 6.0-6.6 |
| Sweep task, 8k, 6 reps × 2 boots | 61.8-65.5 · 7.4-7.9 | 65.9-79.7 · 7.2-7.7 |
| 12 mixed prompts, median speed | 1.00 | **1.10** |

Verbatim reproduction is the best case for a block drafter: 14.5 of a budget of
16. Free-form code lands at 7-9.

## 6. Losslessness

`bench/lossless.py`: 12 prompts (a 2k coding task, a Snake game, a refactor, JSON
extraction, two 32k long-context tasks, six short chat prompts), temperature 0,
seed 0, 512 max tokens, token strings from `logprobs`. Every restart used the
mounted tactic cache, verified byte-identical to the good draw before and after.
2026-09-01. Data: `data/lossless.csv`, hashes and token counts only.

| Comparison | Identical | First divergence, in tokens |
|---|---|---|
| no drafter, run 1 vs run 2 | **12/12** | — |
| drafter on, boot 1 vs boot 2 | 9/12 | 3, 66, 18 |
| drafter on vs no drafter | **3/12** | 84, 52, 24, 54, 6, 4, 8, 126, 16 |
| no drafter idle vs no drafter with 3 concurrent requests | **3/12** | 3, 52, 24, 54, 6, 4, 15, 26, 28 |
| fp8 KV: candidate vs fp8 KV no drafter | 6/12 | — |
| fp8 KV: no drafter idle vs with 3 concurrent requests | 5/12 | — |
| fp8 KV no drafter vs bf16 KV no drafter | **2/12** | — |

Row four is the control, and it is the row that matters. With no drafter
anywhere, merely serving three other requests at the same time flips the same nine
prompts, most at the very same token index (52, 24, 54, 6, 4). Batch size decides
which GEMM tactic runs, the tactic decides the rounding, and the rounding decides
a near-tie. Switching KV dtype with no drafter changes 10 of 12: **the KV dtype is
a larger perturbation than the drafter.**

**Every divergence is a near-tie.** Chosen-against-runner-up logprob gaps at the
divergence points, from the loaded no-drafter run: 0.125, 0.125, 0.125, 0.25,
0.25, 0.375, 0.375, 0.625, 0.875 nats, every one a multiple of 0.125, which is
bf16 resolution at these logit magnitudes. The flips are synonyms. The three
prompts that never flip (JSON extraction, a capital city, a prime list) are the
ones with no near-ties.

**What can honestly be claimed.** Speculative decoding here is lossless in the
only sense that exists for this target: its greedy output is one of the outputs
the same NVFP4 model produces on its own, depending on batch size. A
token-for-token match against a batch-1 baseline is not a property this model has
even with no drafter. A quality claim needs a task benchmark, not a diff, and that
has not been run. No-drafter decode, for reference: 12.6-12.7 tok/s short, 11.4
at 32k, the 5.5-5.9x the drafter buys.

## 7. The fp8 KV cache, measured and not adopted

`--kv-cache-dtype fp8_e4m3`, draft 16, bf16 drafter. Short context at 6 reps, long
context by `ctxsweep.py` at 2 reps and 512 output tokens, 2026-09-01. Comparators
are the bf16-KV numbers from the same evening.

| Prompt tok | bf16 KV: gen / tok-pass / ms-pass | fp8 KV: gen / tok-pass / ms-pass | Prefill bf16 → fp8 |
|---|---|---|---|
| 320 | 63.2 / 7.13 / 113 | 64.7 / 7.13 / 110 | — |
| 8k | 61.8 / 7.38 / 119 | 65.9 / 7.73 / 117 | — |
| 65k | 68.5 / 8.83 / 129 | 67.5 / 8.10 / 120 | 1,846 → 1,620 (−12%) |
| 130k | 52.6 / 8.08 / 154 | **63.2** / 8.15 / **129** | 1,387 → 1,152 (−17%) |
| 260k | 35.9 / 5.98 / 166 | **46.3** / 7.43 / **160** | 929 → 732 (−21%) |

Two things this reverses from the draft-8 era. **Acceptance does not fall with
fp8 KV at draft 16:** the draft-8 measurement (5.03 → 3.90 tokens per pass, the
reason `bfloat16` is in the recipe) does not reproduce, 7.1-8.2 here against
7.1-8.8 on bf16, inside the noise. That claim is now draft-8 only. And **at depth,
fp8 KV is the larger decode win**, +20% at 130k and +29% at 260k, because the
per-pass KV read halves. Short context is a wash, +2 to +7%.

The cost is prefill: −12% at 65k rising to −21% at 260k, cold TTFT 280 → 356 s at
the ceiling. That trade favours an agent which prefills once and decodes many
times, and not a one-shot long prompt.

**Not adopted.** One prompt (`p04_json_extract`) had produced byte-identical text on
every boot that night, with the drafter on, off, under load and with the NVFP4
drafter. With fp8 KV it changed. Adoption needs a task-level quality check first.

## 8. Where a decode pass goes

Torch profiler, captured live with `/start_profile` on a 320-token prompt, one
request of 242 output tokens, 29 passes, analysed with `bench/profpass.py`. bf16
drafter. FlashInfer 0.6.18, cuBLAS 13.1.1.3, CUTLASS 4.7.0, torch 2.13.0+cu130.

| Per pass | ms |
|---|---|
| Wall | 113.0 |
| GPU busy (union of kernels) | 111.3 |
| GPU idle | 1.7 |
| Gaps longer than 0.2 ms | 0.6 |
| Kernels launched | 1,289 |

**The GPU is 98.5% busy. No launch overhead is left to recover.**

| Group | ms | Bytes | Effective GB/s |
|---|---|---|---|
| FP4 GEMM, target MLP (130 calls) | 49.6 | ~8.6 GB | ~173 |
| FP8 GEMM, target attention and GDN projections (128 calls) | 38.1 | ~7.2 GB | ~189 |
| bf16 GEMM, drafter (21 calls) | 17.5 | 3.85 GB | ~220 |
| GDN recurrent update and conv1d | 7.1 | state, small | — |
| norms, quantisation, elementwise | 1.6 | — | — |
| attention (16 layers at 320 tokens) | 0.5 | — | — |

The target's GEMMs run at 75-82% of the measured 231 GB/s bus. At 17 tokens per
verify the FP4 and FP8 kernels are tile-bound, not bandwidth-bound; if both
reached bus speed the pass would shrink by about 19 ms, or 17%. That is the lever
the tactic-cache re-roll already pulled, and what remains needs better small-M
kernels for sm121, not a config flag. The bf16 drafter was 15% of every pass and
already at bus speed, so the only way to shrink it was fewer bytes. That is what
the NVFP4 drafter did: 3.85 → 1.45 GB per pass, 12-23 ms faster. The profile
predicted the win before it was measured.

## 10. Lessons from measuring

Each of these cost real time and changed a number in this document.

- A sweep with no task measures the corpus, not the machine: raw code with no
  instruction collapsed acceptance to 2.1-4.8 tokens per pass and read 40% low.
  Every prompt in `bench/` ends in a real task.
- Three stable readings of one frozen artefact are one reading: three boots
  replaying the same cached draw agreed to 0.9%; a genuine re-tune drew 57.99.
- A green `/v1/models` is not a working server, and `--restart unless-stopped`
  hides a crash loop behind it. `/tokenize` plus `RestartCount` is the check.
- A launch script's default image is a config bug waiting to happen: ours
  defaulted to the pre-#35496 build while every good run passed the tag by hand.
- A same-day A/B with a warm cache is not evidence a config survives a reboot.
- Warm the server before probing it: Prometheus histograms do not exist until
  their first observation, and a cold probe silently disabled a whole sweep's
  cache check.
- A metrics scraper that swallows exceptions returns all zeros, which reads like
  a real measurement of a badly configured server.
- Coding harnesses are too noisy to measure a config change: 30% task-to-task,
  and 3x in prompt tokens for one task. They answer "does it produce working
  code", not "is it faster".
- An in-stream server error can arrive as a one-token reply; count it as an
  error, not a measurement.
