#!/usr/bin/env python3
"""
concbench — aggregate and per-stream generation at N concurrent requests.

Each stream gets its own ~2k-token prefix (unique tag, so the prefix cache
cannot merge them) plus the ctxsweep task, temperature 0, 512 output tokens.
All N start together. Reported per rung:

  aggregate tok/s   total completion tokens / (last stream end - first start)
  per-stream tok/s  median of completion tokens / (wall - TTFT) per stream
  TTFT              median first-token latency under load
  tok/pass          accepted tokens per verification pass (server gauge)

Boot the server with MAX_RUNNING at least as large as the top rung, or the
extra streams queue instead of running. MAX_RUNNING=16 costs maximum context:
the KV pool drops to 238,605 tokens.

Environment:
    SPARK_BASE_URL   default http://localhost:8003/v1
    SPARK_API_KEY    the server's API key
    SPARK_MODEL      default qwen3.8-27b

Usage:
    export SPARK_API_KEY=$(cat ~/models/vllm_api_key.txt)
    python3 bench/concbench.py --label recommended --conc 1,2,4,8,16
"""
import argparse, json, os, pathlib, statistics, threading, time, uuid
from ctxsweep import (API_KEY, BASE_URL, MODEL, RESULTS, TASK, build_prompt,
                      busy, chat, draft_budget, gauge, seed_corpus,
                      spec_counters, spec_delta)

HERE = pathlib.Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_URL)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--label", default="conc")
    ap.add_argument("--conc", default="1,2,4,8,16")
    ap.add_argument("--prefix-tokens", type=int, default=2048)
    ap.add_argument("--out-tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=2, help="repeat each rung; median reported")
    args = ap.parse_args()
    key = API_KEY
    budget = draft_budget(args.base, key)
    seed = seed_corpus(repeat=8)
    out = RESULTS / f"concbench-{args.label}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(exist_ok=True)
    concs = [int(x) for x in args.conc.split(",") if x.strip()]
    nmax = max(concs)
    print(f"endpoint {args.base}   draft budget {budget}   prefix {args.prefix_tokens}   "
          f"out {args.out_tokens}   reps {args.reps}\n")
    # One distinct prompt per stream slot, built once and reused across rungs.
    # The cold prefill is paid in a warm-up pass, so the rungs measure decode.
    prompts = []
    for i in range(nmax):
        p, _ = build_prompt(args.base, args.model, key, args.prefix_tokens,
                            f"conc{i:02d}-{uuid.uuid4().hex[:6]}", seed)
        prompts.append(p + TASK)
    print(f"built {nmax} prompts; warming prefixes ...")
    for p in prompts:
        chat(args.base, args.model, key, p, 8)
    print(f"{'conc':>5} {'agg tok/s':>10} {'per-stream':>11} {'TTFT med':>9} {'tok/pass':>9} {'out tok':>8}")
    print("-" * 60)
    for n in concs:
        rung = []
        for rep in range(args.reps):
            while busy(args.base):
                time.sleep(2)
            c0 = spec_counters(args.base)
            results = [None] * n
            def run(i):
                results[i] = chat(args.base, args.model, key, prompts[i], args.out_tokens)
                results[i]["t_end"] = time.perf_counter()
            t0 = time.perf_counter()
            th = [threading.Thread(target=run, args=(i,)) for i in range(n)]
            for t in th: t.start()
            for t in th: t.join()
            t_end = max(r["t_end"] for r in results)
            ctok = sum(r["ctok"] for r in results)
            agg = ctok / (t_end - t0)
            per = statistics.median(r["ctok"] / max(r["wall"] - r["ttft"], 1e-9) for r in results)
            ttft = statistics.median(r["ttft"] for r in results)
            tpp = gauge(args.base, "sglang:spec_accept_length")
            if tpp is None:
                tpp, _ = spec_delta(c0, spec_counters(args.base))
            rung.append((agg, per, ttft, tpp, ctok))
        agg = statistics.median(r[0] for r in rung); per = statistics.median(r[1] for r in rung)
        ttft = statistics.median(r[2] for r in rung); tpp = rung[-1][3]; ctok = rung[-1][4]
        row = {"label": args.label, "conc": n, "agg_tok_s": round(agg, 2),
               "per_stream_tok_s": round(per, 2), "ttft_med_s": round(ttft, 3),
               "tok_per_pass": tpp, "out_tokens_total": ctok,
               "prefix_tokens": args.prefix_tokens, "out_tokens": args.out_tokens,
               "reps": args.reps}
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{n:>5} {agg:>10.1f} {per:>11.1f} {ttft:>8.2f}s {str(tpp):>9} {ctok:>8}")
    print(f"\nwrote {len(concs)} rows to {out}")


if __name__ == "__main__":
    main()
