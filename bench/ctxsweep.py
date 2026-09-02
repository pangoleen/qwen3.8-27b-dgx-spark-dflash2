#!/usr/bin/env python3
"""
ctxsweep — prompt processing and generation throughput across the context ladder.

The shape Apple-silicon benchmarks publish (0.5k to 256k, prefill and decode at
each length). Running it here makes the two comparable, and it is the right
shape for this project's central finding: a single tok/s number means nothing
without the prompt size attached.

Three things this adds over the format it copies:

  cold vs warm TTFT   the prefix cache made visible per rung
  accepted tokens     accepted tokens per verify pass, the mechanism metric
  saturation          accepted / draft budget, the diagnostic that caught our
                      draft budget being half what it should have been

Prompts are calibrated against the SERVER's own tokenizer, so "8k" means 8k
prompt tokens and not an estimate. Every prompt carries a per-run unique tag,
so the prefix cache cannot serve one measurement from another's leftovers —
except in the warm phase, where that is the point.

Environment:
    SPARK_BASE_URL   default http://localhost:8003/v1
    SPARK_API_KEY    the server's API key
    SPARK_MODEL      default qwen3.8-27b

Usage:
    export SPARK_API_KEY=$(cat ~/models/vllm_api_key.txt)
    python3 bench/ctxsweep.py --label recommended
    python3 bench/ctxsweep.py --lengths 512,1024,2048 --out-tokens 128
"""
import argparse, json, os, pathlib, statistics, time, urllib.request, uuid

HERE = pathlib.Path(__file__).parent
RESULTS = HERE.parent / "results"
LADDER = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

BASE_URL = os.environ.get("SPARK_BASE_URL", "http://localhost:8003/v1")
MODEL = os.environ.get("SPARK_MODEL", "qwen3.8-27b")
API_KEY = os.environ.get("SPARK_API_KEY", "")

TASK = ("\n\n---\nRead the code above. Write a short Python function "
        "`summarise_run(rows)` that takes a list of result dicts and returns a "
        "markdown table of the median, min and max of the 'decode_tok_s' field "
        "per label. Return only the function in one code block.")

# Filler prose for the seed corpus. The corpus must be neutral text this repo
# owns, long enough to grow a 260k-token prompt, and stable between runs so two
# sweeps build the same prompt at the same rung.
PROSE = """
A benchmark is a measurement, and a measurement without its conditions is a
rumour. The conditions that matter on this box are the prompt size, the output
size, the sampler, the number of repeats and the number of boots. Change any
one of them and the headline number moves by more than the config change you
were trying to measure.

Memory bandwidth sets the floor. The machine reads the model weights once per
verification pass, so the fastest a pass can finish is the weight bytes divided
by the bus rate. Everything above that floor is scheduling, and everything
below it is a mistake in the arithmetic.

Speculative decoding pays only when the output is guessable. A drafter proposes
a block of tokens, the target checks them in one pass, and the accepted prefix
is kept. On a verbatim copy task the drafter is right almost every time. On a
free-form answer it is right about half the time. The same server therefore
reports two very different speeds, and both are true.

Caches hide work. A prompt served from the prefix cache costs a lookup instead
of a prefill, so the second reading of the same prompt is not the same
measurement as the first. An agent that keeps one long conversation alive lives
almost entirely in the second case, which is why the cold and the warm numbers
are both reported here rather than averaged into one.

Concurrency is a different clock again. Aggregate throughput rises with the
number of streams while each single stream gets slower, so a repository that
quotes only the aggregate and a repository that quotes only the single stream
can disagree by a factor of four while both are honest. Quote the prompt size,
the stream count, and the clock.

The output of a quantised model is not a fixed function of its input. Batch
size decides which matrix-multiply kernel runs, the kernel decides the rounding,
and the rounding decides any near-tie between two candidate tokens. Two runs of
the same server with the same seed can therefore differ, with or without a
drafter in the loop. That is a property of the arithmetic, not of the drafter.

Write down what you did not measure. A single box on a single day, one boot per
configuration, one task family: those limits belong next to the numbers, not in
a footnote nobody reads.
"""


def seed_corpus(repeat=200):
    """Deterministic filler: this file's own source plus plain prose.

    Repeated to length. No file outside this folder is read, so the corpus is
    the same wherever the repository is checked out.
    """
    block = pathlib.Path(__file__).read_text() + "\n" + PROSE + "\n"
    return block * repeat


def _hdr(key):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    return h


def ntok(base, model, key, text):
    """Token count from the server's own tokenizer, never an estimate."""
    root = base.rsplit("/v1", 1)[0]
    req = urllib.request.Request(root + "/tokenize",
                                 data=json.dumps({"model": model, "prompt": text}).encode(),
                                 headers=_hdr(key))
    d = json.load(urllib.request.urlopen(req, timeout=60))
    return len(d.get("tokens") or d.get("input_ids") or [])


def build_prompt(base, model, key, target, tag, corpus):
    """Grow a prompt to `target` tokens, verified against the server tokenizer."""
    lo, hi = 200, len(corpus)
    body = corpus
    # binary search on characters to land within 1% of the target
    for _ in range(12):
        mid = (lo + hi) // 2
        cand = f"[run {tag}]\n" + corpus[:mid]
        n = ntok(base, model, key, cand)
        if abs(n - target) <= max(8, target // 100):
            return cand, n
        if n < target:
            lo = mid
        else:
            hi = mid
        body = cand
    return body, ntok(base, model, key, body)


def chat(base, model, key, prompt, max_tokens, timeout=3600, temperature=0.0):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False}}
    req = urllib.request.Request(base + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=_hdr(key))
    t0 = time.perf_counter()
    ttft, usage, chunks = None, {}, []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            ln = raw.decode().strip()
            if not ln.startswith("data: "):
                continue
            if ln[6:] == "[DONE]":
                break
            try:
                o = json.loads(ln[6:])
            except json.JSONDecodeError:
                continue
            if o.get("error") or o.get("object") == "error":
                # The server returns limit and validation errors inside the
                # stream. Without this they are counted as a 1-token completion.
                raise RuntimeError(f"server error in stream: {str(o.get('error') or o.get('message'))[:200]}")
            if o.get("usage"):
                usage = o["usage"]
            for c in o.get("choices", []):
                d = c.get("delta", {}).get("content")
                if d:
                    chunks.append(d)
                    if ttft is None:
                        ttft = time.perf_counter() - t0
    wall = time.perf_counter() - t0
    return {"wall": wall, "ttft": ttft or wall,
            "ptok": usage.get("prompt_tokens", 0),
            "ctok": usage.get("completion_tokens", 0),
            "cbytes": len("".join(chunks).encode()), "cchars": len("".join(chunks))}


def gauge(base, name):
    try:
        b = urllib.request.urlopen(base.rsplit("/v1", 1)[0] + "/metrics", timeout=10).read().decode()
    except Exception:
        return None
    for ln in b.splitlines():
        if ln.startswith(name + "{") or ln.startswith(name + " "):
            try:
                return float(ln.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                pass
    return None


def busy(base):
    return ((gauge(base, "sglang:num_running_reqs") or 0) + (gauge(base, "sglang:num_queue_reqs") or 0)
            + (gauge(base, "vllm:num_requests_running") or 0) + (gauge(base, "vllm:num_requests_waiting") or 0))


def spec_counters(base):
    """vLLM reports speculative decoding as monotonic counters. A per-rung delta
    gives tokens per verification pass exactly. Returns None on SGLang, whose
    gauge path is used instead."""
    d = gauge(base, "vllm:spec_decode_num_drafts_total")
    if d is None:
        return None
    return {"drafts": d,
            "draft_tokens": gauge(base, "vllm:spec_decode_num_draft_tokens_total") or 0.0,
            "accepted": gauge(base, "vllm:spec_decode_num_accepted_tokens_total") or 0.0}


def spec_delta(before, after):
    """(tokens per pass, draft budget) from two vLLM counter snapshots."""
    if not before or not after:
        return None, None
    drafts = after["drafts"] - before["drafts"]
    if drafts <= 0:
        return None, None
    acc = after["accepted"] - before["accepted"]
    budget = (after["draft_tokens"] - before["draft_tokens"]) / drafts
    return round(1.0 + acc / drafts, 3), round(budget, 2)


def draft_budget(base, key):
    for path in ("/get_server_info", "/server_info"):
        try:
            req = urllib.request.Request(base.rsplit("/v1", 1)[0] + path, headers=_hdr(key))
            info = json.load(urllib.request.urlopen(req, timeout=15))
        except Exception:
            continue
        for scope in (info, info.get("server_args") or {}):
            v = (scope or {}).get("speculative_num_draft_tokens")
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_URL)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--label", default="sweep")
    ap.add_argument("--lengths", default=",".join(str(x) for x in LADDER))
    ap.add_argument("--out-tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--skip-warm", action="store_true",
                    help="skip the cache-hit phase (halves the run time)")
    args = ap.parse_args()

    key = API_KEY
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    budget = draft_budget(args.base, key)
    seed = seed_corpus()

    out = RESULTS / f"ctxsweep-{args.label}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(exist_ok=True)
    print(f"endpoint {args.base}   draft budget {budget}   out_tokens {args.out_tokens}\n")
    print(f"{'ctx':>8} {'prompt tok':>11} {'prefill t/s':>12} {'gen t/s':>9} "
          f"{'tok/pass':>9} {'sat':>5} {'cold TTFT':>10} {'warm TTFT':>10}")
    print("-" * 82)

    # Prompt plus output must fit the context limit. The server rejects the
    # request (HTTP 400) rather than truncating, so leave a margin for the
    # chat template.
    CTX_LIMIT = 262144
    lengths = [min(L, CTX_LIMIT - args.out_tokens - 2048) for L in lengths]

    rows = []
    for L in lengths:
        while busy(args.base):
            time.sleep(3)
        tag = uuid.uuid4().hex[:8]
        try:
            prompt, ptok = build_prompt(args.base, args.model, key, L, tag, seed)
            # Without a task on the end the model free-continues a code dump and
            # acceptance collapses to 2-4 tokens per pass. The sweep then
            # measures how predictable the corpus is, not the machine.
            prompt = prompt + TASK
        except Exception as e:
            print(f"{L:>8}  prompt build failed: {e}")
            continue

        cold = None
        gens, ttfts = [], []
        c0 = spec_counters(args.base)
        for r in range(args.reps):
            res = chat(args.base, args.model, key, prompt, args.out_tokens, temperature=args.temperature)
            if res["ctok"] < 8:
                print(f"{L:>8}  only {res['ctok']} completion tokens; skipping")
                break
            if cold is None:
                cold = res           # first pass is the cold one
            dt = max(res["wall"] - res["ttft"], 1e-9)
            gens.append(res["ctok"] / dt)
            ttfts.append(res["ttft"])
        if not gens:
            continue

        warm = None
        if not args.skip_warm:
            w = chat(args.base, args.model, key, prompt, args.out_tokens, temperature=args.temperature)
            warm = w["ttft"]

        tpp = gauge(args.base, "sglang:spec_accept_length")
        if tpp is None:
            tpp, vb = spec_delta(c0, spec_counters(args.base))
            budget = budget or (vb + 1 if vb else None)   # vLLM: budget tokens plus 1 bonus
        sat = (tpp / budget) if (tpp and budget) else None
        row = {"label": args.label, "temperature": args.temperature, "ctx_target": L,
               "prompt_tokens": cold["ptok"],
               "prefill_tok_s": round(cold["ptok"] / cold["ttft"], 1),
               "gen_tok_s": round(statistics.median(gens), 2),
               "tok_per_pass": tpp, "saturation": round(sat, 3) if sat else None,
               "cold_ttft_s": round(cold["ttft"], 3),
               "warm_ttft_s": round(warm, 3) if warm else None,
               "out_tokens": cold["ctok"], "reps": len(gens),
               "prompt_bytes": len(prompt.encode()), "prompt_chars": len(prompt),
               "gen_bytes": cold["cbytes"], "gen_chars": cold["cchars"],
               "prefill_bytes_s": round(len(prompt.encode()) / cold["ttft"], 1),
               "gen_bytes_s": round(cold["cbytes"] / max(cold["wall"] - cold["ttft"], 1e-9), 1),
               "tpot_ms": round(1000.0 / statistics.median(gens), 2) if gens else None,
               "total_s": round(cold["wall"], 2)}
        rows.append(row)
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{L:>8} {row['prompt_tokens']:>11} {row['prefill_tok_s']:>12.1f} "
              f"{row['gen_tok_s']:>9.2f} {str(tpp):>9} "
              f"{(f'{100*sat:.0f}%' if sat else '-'):>5} "
              f"{row['cold_ttft_s']:>9.2f}s "
              f"{(f'{warm:.2f}s' if warm else '-'):>10}")

    print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
