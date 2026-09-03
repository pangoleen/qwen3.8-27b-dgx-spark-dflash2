# Qwen3.8-27B on one DGX Spark (GB10)

By Paolo Rosson, [@redp314 on X](https://x.com/redp314), where the results and
follow-ups are posted first.

A serving recipe for a single DGX Spark: SGLang, NVFP4 weights, and a DFlash2
speculative drafter at a draft budget of 16. It ships the engine image build, the
launch script, the six benchmarks that produced every number below, the rows
behind them as CSV, and the kept FlashInfer autotune draws, the fix for a boot lottery worth 16%.

| Single stream, 320 tokens to 65k of context | 16 concurrent streams | 32 concurrent streams | Accepted tokens per verify pass |
|---|---|---|---|
| **64-78 tok/s**, two boots | **360 tok/s** aggregate, 27 per stream | **387 tok/s** aggregate, 15 per stream (32-seat profile) | 6.8-8.3 of a budget of 16 |

Conditions: a coding task on real source, 512 output tokens, temperature 0,
thinking off, decode only (time to first token excluded), median of 4 runs per
rung, two boots; the concurrency figures are distinct 2k-token code prefixes,
512 output tokens, median of 2, one boot per profile. Past 65k it falls: 56 at 130k, 38-41 at 260k. Everything
else, with its conditions, is in [RESULTS.md](RESULTS.md).

**Replicated from this repository alone, 2026-09-02:** a fresh copy of the repo
on the same box, image built from `image/` (same image ID as the one measured on,
`bd783c5f0356`), an empty tactic cache with the shipped draws installed by the
README command, `serve.sh` with its defaults. Boot 210 s, no `kept eager`, KV
pool 414,647 tokens, and the shipped sweep gave 79.0 tok/s at 311 prompt tokens
and 70.0 at 8,318, inside the two-boot range above.

![Nine-panel context sweep](charts/ctxsweep-27b.png)

## Requirements

| Item | Value | Source |
|---|---|---|
| Box | NVIDIA DGX Spark, GB10 Grace Blackwell, sm_121, aarch64 | the box |
| Memory | 128 GB LPDDR5X unified, ~273 GB/s spec | vendor spec |
| Effective bandwidth | ~231 GB/s, and 235 GB/s from the per-pass slope | measured here |
| Engine | SGLang: the pinned day-0 image plus upstream PR #35371 and **#35496**, built from `image/` in one command | `image/NOTICE.md` |
| Target | `RadixArk/Qwen3.8-27B-NVFP4`, packed FP4 lm_head, `modelopt_mixed` | checkpoint card |
| Drafter | `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal`, 1.45 GB read per pass | boot log, profiler |
| Model shape | 64 layers, 16 full attention and 48 Gated DeltaNet | checkpoint config |
| Context | 262,144 tokens; the practical ceiling is ~261,888, because the output comes out of the same allowance | arithmetic |
| Disk | 21 GB target + 1.5 GB drafter in the HF cache, ~39 GB engine image, <1 MB tactic cache | `du` on our box |
| Docker + NVIDIA Container Toolkit | required (`docker run --gpus all`); every cache operation must run inside a container | measured the hard way |

PR #35496 folds the DFlash2 path selector into the draft CUDA graph on a packed
FP4 lm_head. Without it the selector runs eager and throughput drops; `serve.sh`
checks the boot log and says which one you got.

**Weights are not included** and carry Qwen's own licence. Pin both checkpoints
by `--revision`. The engine image is built locally from a base pinned by digest
plus the eight patched files in `image/`:

```bash
docker build -t qwen38-27b-sglang-dflash2-sm121:0.3.0 -f image/Dockerfile image/
```

## Weights

The server runs with `HF_HUB_OFFLINE=1` and never downloads anything. Put both
checkpoints in the cache `serve.sh` mounts (`HF_CACHE`, default `~/models/hf`,
which must contain a `hub/` directory in the standard Hugging Face layout):

```bash
pip install -U huggingface_hub          # gives the `hf` CLI (older installs call it huggingface-cli)
export HF_HOME=$HOME/models/hf
hf download RadixArk/Qwen3.8-27B-NVFP4 --revision 554ebba9b5f1b79dc11246341960360e6ef05ef4
hf download maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal --revision bd7a934213c47a9e7ef69eef36bb3325f47fd1f1
```

Those are the snapshots every number here was measured on. `.env.sample`
exports them as `REVISION` and `DRAFT_REVISION`, and `serve.sh` passes them to
the engine, so a SHA-only download (which writes no `refs/main`) still resolves. One trap:
a partial download can leave `hub/models--RadixArk--Qwen3.8-27B-NVFP4/refs/main`
pointing at an incomplete snapshot (ours drifted to one holding only tokenizer
files), and the offline server then crash-loops on "Can't load image processor".
`cat refs/main` must name a snapshot directory that holds the safetensors.

## Quickstart

```bash
git clone https://github.com/pangoleen/qwen3.8-27b-dgx-spark-dflash2 && cd qwen3.8-27b-dgx-spark-dflash2
docker build -t qwen38-27b-sglang-dflash2-sm121:0.3.0 -f image/Dockerfile image/   # ~1 min on top of the base pull
mkdir -p ~/models && openssl rand -hex 32 > ~/models/vllm_api_key.txt && chmod 600 ~/models/vllm_api_key.txt
cp .env.sample .env && source .env   # exports SPARK_* and the two checkpoint revisions
./serve.sh                       # 170-240 s; prints "ready after N0s", then tokenize=200 and restarts=0
export SPARK_API_KEY=$(cat ~/models/vllm_api_key.txt)
curl -s http://localhost:8003/v1/chat/completions \
  -H "Authorization: Bearer $SPARK_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Write a haiku about GPUs."}],"max_tokens":64}'
./stop.sh                        # when done; the restart policy needs an explicit stop
```

`serve.sh` waits for health, then checks two things a green `/v1/models` does not
prove: that `/tokenize` answers 200, which needs a live scheduler, and that the
container's restart count is still zero.

## What you get, and under what conditions

**Generation across the context ladder.** `bench/ctxsweep.py`, NVFP4 drafter,
bf16 KV, 512 output tokens, 4 reps per rung, temperature 0, `MAX_RUNNING=4`, two
boots on 2026-09-02. Five of the ten rungs; the rest are in RESULTS.md §1.

| Prompt tokens | Prefill tok/s | Generation, boot 1 | Generation, boot 2 | Accepted tok/pass |
|---|---|---|---|---|
| 324 | 1,657 | 72.7 | 75.6 | 6.88 |
| 8,159 | 2,506 | 73.9 | 75.5 | 7.53 |
| 65,842 | 1,811 | 66.8 | 63.8 | 8.12 |
| 130,656 | 1,372 | 55.9 | 56.4 | 7.92 |
| 257,334 | 928 | 38.2 | 41.3 | 6.83 |

Generation holds 64-78 tok/s from 320 tokens to 65k, then falls. Only 16 of the 64
layers pay for context depth; the other 48 carry constant-size recurrent state.
Boot to boot the same rungs differ by a median 4.3% and at worst 8.1%, which is
why the honest headline is "about 70 tok/s to 65k", not one rung to three digits.

**Chat is slower than code, and that is the drafter, not the setup.** The
64-78 tok/s above is a coding task on real source. On mixed chat prompts the
same server does **~30 tok/s**: the block drafter accepts ~3 tokens per verify
pass on prose against 7-9 on code, at the same ~100 ms per pass. Measured with
the Inference Atlas harness on this box (30.4 tok/s, 3.1 accepted) and
reproduced independently on another Spark (33 tok/s, ~3 accepted, 102 ms ITL).
Prefill is unaffected. Every published drafter for this model behaves the same
way on chat; a smaller draft budget does not help (8 is +3%, 4 is −12%).

**Concurrency.** `bench/concbench.py`: every stream gets its own 2k-token code
prefix (pre-filled once), 512 output tokens, temperature 0, all streams started
together, median of 2 runs per rung. Two profiles, one boot each.

| Streams | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 16-seat profile (`MAX_RUNNING=16`, mem 0.65): aggregate tok/s | 81.6 | 119.3 | 199.4 | 276.0 | **359.5** | — |
| per stream / median TTFT | 85.0 / 0.14 s | 66.1 / 0.27 s | 57.5 / 0.31 s | 42.7 / 0.46 s | 26.8 / 0.70 s | — |
| 32-seat profile (`MAX_RUNNING=32`, mem 0.80): aggregate tok/s | 67.6 | 115.1 | 168.8 | 224.1 | 337.4 | **387.2** |
| per stream / median TTFT | 70.8 / 0.14 s | 66.2 / 0.27 s | 51.2 / 0.31 s | 37.4 / 0.38 s | 26.1 / 0.80 s | 15.3 / 1.09 s |

Accepted tokens per pass stay at 7-9 all the way up. The two profiles are
different boots and are drawn as two lines, not one curve: below 16 streams the
32-seat boot runs 8-19% under the 16-seat one. A 32-seat boot at the recipe's
0.65 memory share is a trap: the KV pool collapses to 49,958 tokens and every
rung loses (213 tok/s at 32); the 0.80 share gives it 173,120. Rows for all
three boots are in `data/concbench.csv`.

![Concurrency](charts/concurrency.png)

**Three recipes, same box, same prompts.** The drafter swap is the adoption;
fp8 KV is measured but held back (RESULTS.md §7).

![Three recipes compared](charts/three-recipes.png)

**Counting and methodology, read before comparing.** Generation is decode only:
completion tokens divided by (wall time minus time to first token), so prefill is
never counted as decode. Prefill is prompt tokens divided by the cold time to first
token. Accepted tokens per pass is the server's own `sglang:spec_accept_length`
gauge. Prompt sizes are calibrated against the server's own tokenizer, so "8k"
means 8,192 prompt tokens and not an estimate. Thinking is off, temperature is 0
unless a table says otherwise, and every table states its reps and its boots. Most
are one boot. **A boot is the largest source of variance here, and one boot is not
a distribution.**

## The FlashInfer tactic cache (our one non-obvious win)

FlashInfer autotunes: at boot it times several candidate kernels for each GEMM
shape and keeps the fastest. Timing noise means different boots pick different
tactics, so the same config measures differently every time. That is the boot
lottery, and the usual fix in this field is to turn autotuning off and take the
determinism instead. The cause is simpler than that. FlashInfer's own
documentation says autotuning results live in memory and are lost when the
process exits, and that without a volume mount the cache directory may not
survive a container restart. We ran containerised and never mounted it. **Every
boot re-timed every tactic from scratch.**

Mount the cache, leave autotuning **on**, and keep a good draw:

```bash
# the line in serve.sh that does it
-v "${SGLANG_CACHE}":/root/.cache/sglang     # default ~/sglang-cache
```

- After mounting, fresh decode on the bf16-drafter recipe went **60.85 → 70.6
  tok/s** (median of five boots replaying the same draw, within 1.3%).
- Draws cluster into families at 6.6 / 8.55 / 8.75 / 9.68 accepted tokens per
  pass. Roughly one in three is good. A cleared-cache re-tune drew **57.99**
  tok/s; the draw we kept measured **70.4** on the boot that produced it. Clear,
  boot, measure, and keep the first draw above 68 tok/s.
- Against a median boot the kept draw is worth **+16%** on fresh decode.
- Only **6 of 35** tactic slots mattered at batch size 1 on the bf16-drafter draw
  (the NVFP4-drafter draw has 49 entries; the same reasoning applies). Lookup rounds the batch
  dimension **up to the next power of two**, so one lucky draw generalises across
  batch sizes. You are picking a kernel plan, not fitting the test set.
- `AUTOTUNE=0` drops `--disable-flashinfer-autotune` back in and the mounted draw
  is ignored. The default here is `AUTOTUNE=1`.

`data/tactic-cache/` holds the draws these numbers were taken on, in the layout
the mount expects. FlashInfer keys the cache by a hash of the tuned
configuration, so there are two directories: `772fea630fdf214a` was written by
the bf16-drafter boots and `6affbca9eddbd34b` by the NVFP4-drafter boots that
`serve.sh` defaults to. Install both, through a container, because the mounted
tree is root-owned and a host-side `cp` fails silently on the one file that
matters (that is how the first good draw was lost):

```bash
docker run --rm -v "$HOME/sglang-cache":/c -v "$PWD/data/tactic-cache":/d:ro busybox sh -c '
  for h in 772fea630fdf214a 6affbca9eddbd34b; do
    mkdir -p /c/flashinfer/autotune/0.6.18/sm121/$h &&
    cp /d/$h/rank_tp0_pp0_dp0.json /c/flashinfer/autotune/0.6.18/sm121/$h/ &&
    chmod 600 /c/flashinfer/autotune/0.6.18/sm121/$h/rank_tp0_pp0_dp0.json
  done; ls -la /c/flashinfer/autotune/0.6.18/sm121/*/'
```

`0.6.18` is the FlashInfer version in the image and `sm121` the GB10. If a boot
creates a third hash directory, your configuration differs from ours and that
boot re-tuned; measure it, and keep it only if it is a good draw. These files are
a kernel plan for this image on this silicon, a starting point, not a guarantee.

## Never quote a throughput number without the prompt size

| Workload | Prompt | Streams | Aggregate tok/s |
|---|---|---|---|
| Toy ladder (draft 8, `MAX_RUNNING=16`) | 31 tokens | 16 | 510-536 |
| sharegpt (`sglang.benchmark.serving`, 320 prompts) | ~439 average | 16 | 187 |
| Realistic agent context | 15,300 | 8 | **47** |
| Realistic agent context | 15,300 | 16 | does not fit the token pool |

**A 10x spread from prompt size alone, on one box.** Speculative decoding pays
when the output is guessable and prefill pays per token, so the blend of prompt
and output decides the number far more than the silicon does. On sharegpt,
achieved concurrency was 15.51 of 16 with a mean TTFT of 2.0 s, so the server was
saturated and the drop is real work, not idle time. The 510 is a 31-token toy at
draft 8. The number worth quoting is **47 tok/s at 8 streams on 15.3k prompts**.

## Tuning

Every knob in `serve.sh` is an environment variable, so a sweep can change one
thing without editing the file.

| Variable | Default | Effect | Measured note |
|---|---|---|---|
| `DRAFT` | `maurienne-ai/…-DFlash2-NVFP4-RTNcal` | the speculative drafter | 1.45 GB per pass instead of 3.85, +5-10% generation |
| `DRAFT_QUANT` | `modelopt_fp4` | names the drafter's quantisation scheme | required for that checkpoint; the same path the target runs |
| `DRAFT_TOKENS` | `16` | tokens per verify pass | 8 → 16 is +69.3% edit, +10.5% fresh; 24 is past the knee |
| `KV_DTYPE` | `bfloat16` | KV cache dtype | `fp8_e4m3` is +20-29% past 130k and −12-21% prefill; not adopted |
| `MAX_RUNNING` | `4` | concurrent requests | 4 keeps a 413,460-token KV pool; 16 buys 360 tok/s aggregate and cuts it to 238,605; 32 needs `MEM_FRACTION=0.80` (pool 173,120, 387 tok/s) or it collapses to 49,958 |
| `AUTOTUNE` | `1` | keeps FlashInfer autotuning on | required for the mounted tactic cache to be used |
| `SGLANG_CACHE` | `~/sglang-cache` | host path for the tactic cache | the whole of the section above |
| `HF_CACHE` | `~/models/hf` | host path of the Hugging Face cache, mounted read-only-in-spirit with `HF_HUB_OFFLINE=1` | see Weights |
| `REVISION`, `DRAFT_REVISION` | empty | pin the target / drafter snapshot | the SHAs under Weights |
| `MEM_FRACTION` | `0.65` | static memory fraction | pairs with `MAX_RUNNING=4` on 128 GB unified |
| `CTX` | `262144` | context length | prompt plus output share it |
| `SPEC` | `1` | `0` launches the same target with no drafter | the losslessness baseline, 12.6-12.7 tok/s short |
| `IMAGE` | `…-dflash2-sm121:0.3.0` | engine image | 0.2.0 runs the selector eager; the script warns |
| `CPUSET` | `5-9,15-19` | pins to the big cores | inherited, never measured here; set `CPUSET=""` to unpin |
| `DISABLE_PREFILL_GRAPH` | `1` | disables prefill CUDA graphs | inherited; untested under DFlash2, where the drafter works during prefill |
| `DRAFT_WINDOW` | empty | draft KV window | **refused**; it parses, boots, then kills the scheduler |
| `SPARK_API_KEY_FILE` | `~/models/vllm_api_key.txt` | where the key is read from | never on a command line |

## Reproducing these numbers

All six benchmarks read `SPARK_BASE_URL` (default `http://localhost:8003/v1`),
`SPARK_API_KEY` and `SPARK_MODEL` (default `qwen3.8-27b`) from the environment.
They write one jsonl per run into `results/`, which is git-ignored.

```bash
export SPARK_API_KEY=$(cat ~/models/vllm_api_key.txt)
python3 bench/ctxsweep.py  --label recommended --out-tokens 512 --reps 4
python3 bench/concbench.py --label recommended --conc 1,2,4,8,16   # boot with MAX_RUNNING>=16
python3 bench/plotconc.py data/concbench.csv --out charts/concurrency.png
python3 bench/plotx.py data/ctxsweep-recommended.csv --panels all \
        --out charts/ctxsweep-27b.png
python3 bench/plotx.py data/ctxsweep-three-recipes.csv --panels compare \
        --out charts/three-recipes.png
```

`bench/` needs only the standard library plus matplotlib for the charts
(`pip install -r bench/requirements.txt`).

The sweep is the long one: its ceiling rung alone pays a 277-second cold prefill
before a single token comes out, once per repeat. Use `--skip-warm` and a shorter
`--lengths` list for a quick check. `bench/plotx.py` rebuilds every chart in
`charts/` from `data/`; a chart that cannot be rebuilt that way does not ship.

## Thinking and tool calling

The server starts with `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`
and thinking **off** by default (`enable_thinking: false` in the chat template
kwargs). Every number in this repository is thinking off. Turn it on per request
with `"chat_template_kwargs": {"enable_thinking": true}`; expect lower accepted
tokens per pass and more output tokens, which is a heavier workload, not a slower
server. Tool calls come back in the OpenAI `tool_calls` shape.

## Using the API

OpenAI-compatible on `:8003`. Streaming with `stream_options.include_usage` is
what every script in `bench/` uses; `usage.completion_tokens` divided by the
time after the first token is the decode figure quoted here.

```bash
curl -s http://localhost:8003/v1/chat/completions \
  -H "Authorization: Bearer $SPARK_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Refactor this function: ..."}],
       "max_tokens":512,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
```

## Logs and troubleshooting

- `docker logs -f sglang38-dflash2`: boot takes 170-240 s; `max_total_num_tokens`
  in the log is the KV pool (413,460 at `MAX_RUNNING=4`).
- A green `/v1/models` before the scheduler is live is normal; wait for
  `/tokenize` to return 200, which `serve.sh` does for you.
- `docker inspect --format '{{.RestartCount}}' sglang38-dflash2` must be 0. A
  non-zero count behind `--restart unless-stopped` is a crash loop, not a server.
- `kept eager (reason=quantized lm_head)` in the log means the image lacks PR
  #35496; rebuild from `image/`.
- `Input length ... exceeds the maximum allowed length` is the KV pool, not the
  context window: lower `MAX_RUNNING` or raise `MEM_FRACTION`.
- Metrics are on `/metrics`; `sglang:spec_accept_length` is the acceptance gauge
  the tables quote.

## Traps

The ones a replicator can hit. Lessons from measuring are in RESULTS.md.

- **A quantised drafter can load cleanly and silently stop speculating.** The
  GPTQ build logged a healthy `quant=compressed-tensors` and produced exactly
  1.0 tokens per pass. Watch `saturation` on `/metrics`, not tok/s; 6% means no
  speculation at all. The NVFP4 drafter in `serve.sh` is the one that works.
- **Autotuning without a mounted cache re-times every tactic on every boot**, and
  different boots land 20% apart. Mount `SGLANG_CACHE`, install the shipped draws,
  and keep autotune on.
- **Every tactic-cache operation must go through a container.** The files are
  root-owned; a host-side `cp` produces an empty directory that looks like a backup.
- **`DRAFT_WINDOW` is refused on purpose.** The flag parses and boots, then kills
  the scheduler on the first batch; the drafter already has a 2,048-token sliding
  window from its config, so it would not have helped.
- **`MAX_RUNNING=16` costs context.** The KV pool drops from 413,460 tokens to
  238,605 and inputs above that are refused with HTTP 400.
- **Greedy output is not stable at this precision, drafter or no drafter.** Batch
  size alone flips 9 of 12 greedy prompts at the same token positions the drafter
  flips. Do not test losslessness with a token diff; see RESULTS.md §6.

## Set it up with an agent

`AGENT_SETUP.md` is a self-contained prompt. Paste it into a coding agent with
shell access on the Spark (Claude Code, Codex, or similar) and it will download
the weights, build the image, install the tactic draws, launch, verify, and run
one rung of the sweep, reporting each step. It never prints the API key.

## What's in here

```
serve.sh                        launch the server           every knob above
stop.sh                         stop it (restart policy needs an explicit stop)
.env.sample                     SPARK_* variables for bench/
bench/ctxsweep.py               context ladder, the headline instrument
                                                            --lengths --reps --temperature
bench/concbench.py              concurrency ladder          --conc, needs MAX_RUNNING>=N
bench/accfix.py                 fixed-output acceptance     --lengths, isolates the drafter
bench/lossless.py               greedy token diff           --compare, and SPEC=0 for the arm
bench/profpass.py               per-pass profiler breakdown /start_profile trace as input
bench/plotx.py                  rebuild the sweep charts    --panels all|three|compare
bench/plotconc.py               rebuild the concurrency chart from data/concbench.csv
data/ctxsweep-recommended.csv   the sweep, T=0 and T=1 rows
data/ctxsweep-three-recipes.csv three recipes at each rung
data/lossless.csv               one row per prompt per arm; hashes, no completion text
data/concbench.csv              one row per (profile, streams); three boots
data/tactic-cache/<hash>/       the kept autotune draws     SGLANG_CACHE, see the section above
charts/*.png                    regenerated from data/ by bench/plotx.py
RESULTS.md                      every table, with its conditions, and the measuring lessons
AGENT_SETUP.md                  paste into a coding agent on the Spark to do the whole setup
image/                          Dockerfile + the 8 patched SGLang files  docker build
```

## Limitations

- **One box, one weekend, mostly one boot per configuration.** The boot-to-boot
  check in RESULTS.md §1 is two boots and already shows a 4.3% median difference.
- **Quality is not measured.** Losslessness here is a token diff, and it shows
  that batch load alone changes greedy output. A real quality claim needs a task
  benchmark, and none has been run.
- **fp8 KV, `CPUSET` and `DISABLE_PREFILL_GRAPH` are not settled.** The first is
  measured and held back; the other two are inherited and untested here.
- **The agent end-to-end number is missing on purpose.** Our coding harnesses
  cannot resolve a 10% change through 30% task noise, so no such number is
  published.
- **Weights are not included** and carry Qwen's own licence, including its
  revenue and monthly-user clause. Read it before you deploy this.
- The engine image is a local overlay on a base pinned by digest; `image/` has
  the build. It is frozen at the day-0 SGLang plus two PRs, not tracking main.

## Credits

SGLang, and upstream PRs #35371 and #35496. RadixArk for the NVFP4 target
checkpoint. The DFlash2 authors for the drafter design, and maurienne-ai for the
NVFP4 drafter build this recipe defaults to.

## License

MIT, see [LICENSE](LICENSE). Model weights are not covered by it.

Questions and results from your own box: [@redp314](https://x.com/redp314).
