#!/usr/bin/env python3
"""
lossless — check what speculative decoding changes in the output.

Every speed claim assumes the drafter is lossless: greedy output with
speculation on should match greedy output with speculation off. This script
does not itself turn speculation on or off — that is a server restart — it runs
one fixed prompt set against each server and diffs the results.

Run mode records, per prompt: the completion text, its sha256 and, when the
server returns them, the per-token strings from logprobs. Compare mode pairs two
result files by prompt id and reports where they first diverge.

Run the batch-load control as well (the same server, idle, then under three
concurrent requests, with no drafter anywhere). Without it a divergence between
the drafter arms cannot be told apart from the arithmetic noise that batch size
alone produces.

The prompt set uses fixed tags, not random ones, so the same prompt text is sent
in every arm. Helpers and the seed corpus come from ctxsweep.py in this folder.

Environment:
    SPARK_BASE_URL   default http://localhost:8003/v1
    SPARK_API_KEY    the server's API key
    SPARK_MODEL      default qwen3.8-27b

Usage:
    export SPARK_API_KEY=$(cat ~/models/vllm_api_key.txt)
    python3 bench/lossless.py --label spec-on
    SPARK_BASE_URL=http://localhost:8002/v1 python3 bench/lossless.py --label spec-off
    python3 bench/lossless.py --compare results/lossless-spec-on-*.jsonl \\
                                        results/lossless-spec-off-*.jsonl
"""
import argparse, hashlib, json, os, pathlib, time, urllib.request
from ctxsweep import (API_KEY, BASE_URL, MODEL, RESULTS, TASK, _hdr,
                      build_prompt, busy, seed_corpus)

HERE = pathlib.Path(__file__).parent

REFACTOR_SNIPPET = '''def classify(x):
    if x is not None:
        if x >= 0:
            if x == 0:
                result = "zero"
            else:
                if x < 10:
                    result = "small positive"
                else:
                    result = "large positive"
        else:
            result = "negative"
    else:
        result = "none"
    return result
'''

JSON_EXTRACT_TEXT = (
    "Invoice #A-4471, issued 2026-03-14 to Marlowe Fabrication Ltd. "
    "Line items total $2,340.00 before tax. Status: overdue since 2026-04-01. "
    "Contact: billing@marlowefab.example."
)


def chat(base, model, key, prompt, max_tokens, temperature, seed, timeout=900):
    """Like ctxsweep.chat(), plus logprobs, so a divergence can be read as a
    near-tie (small gap to the runner-up) or a real flip (large gap). If the
    server omits them, `tokens` comes back empty and compare falls back to a
    character diff."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
            "stream": True, "stream_options": {"include_usage": True},
            "logprobs": True, "top_logprobs": 2,
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False}}
    req = urllib.request.Request(base + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=_hdr(key))
    t0 = time.perf_counter()
    ttft, usage, chunks, tokens, top2 = None, {}, [], [], []
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
            if o.get("usage"):
                usage = o["usage"]
            for c in o.get("choices", []):
                d = c.get("delta", {}).get("content")
                if d:
                    chunks.append(d)
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                lp = c.get("logprobs") or {}
                for item in lp.get("content") or []:
                    if "token" in item:
                        tokens.append(item["token"])
                        tl = item.get("top_logprobs") or []
                        top2.append([[t.get("token"), t.get("logprob")] for t in tl[:2]]
                                    or [[item["token"], item.get("logprob")]])
    wall = time.perf_counter() - t0
    return {"wall": wall, "ttft": ttft or wall,
            "ptok": usage.get("prompt_tokens", 0),
            "ctok": usage.get("completion_tokens", 0),
            "text": "".join(chunks), "tokens": tokens, "top2": top2}


def default_prompts(base, model, key):
    """12 prompts: short chat, code tasks, and two built long by build_prompt().
    Tags are fixed per prompt id, so the same text is produced every run."""
    corpus = seed_corpus()
    prompts = []

    p2k, _ = build_prompt(base, model, key, 2000, "lossless-p01", corpus)
    prompts.append({"id": "p01_ctxsweep_2k", "text": p2k + TASK})

    prompts.append({"id": "p02_snake", "text":
        "Write a complete, playable Snake game in a single HTML file using "
        "canvas and vanilla JavaScript. Arrow keys move the snake, eating "
        "food grows it by one, and hitting a wall or itself ends the game "
        "with a visible restart button. Return only the code in one HTML "
        "code block."})

    prompts.append({"id": "p03_refactor", "text":
        "Refactor the function below for clarity. Keep its behavior "
        "identical for every input. Return only the refactored function in "
        "one code block.\n\n```python\n" + REFACTOR_SNIPPET + "```"})

    prompts.append({"id": "p04_json_extract", "text":
        "Extract the following fields from the text below as a single JSON "
        "object with keys invoice_id, date, company, total_usd, status: \n\n"
        + JSON_EXTRACT_TEXT + "\n\nReturn only the JSON object, nothing else."})

    p32k_a, _ = build_prompt(base, model, key, 32000, "lossless-p05", corpus)
    prompts.append({"id": "p05_long_32k_summary", "text": p32k_a + TASK})

    p32k_b, _ = build_prompt(base, model, key, 32000, "lossless-p06", corpus)
    prompts.append({"id": "p06_long_32k_riskscan", "text": p32k_b +
        "\n\n---\nName the single biggest correctness risk in the code "
        "above, in one paragraph. Do not write code."})

    short = [
        ("p07_capital", "What is the capital of France? Answer in one word."),
        ("p08_recursion", "Explain recursion to a beginner in exactly three sentences."),
        ("p09_haiku", "Write a haiku about debugging code."),
        ("p10_primes", "List three prime numbers greater than 100, comma separated."),
        ("p11_convert", "Convert 98.6 Fahrenheit to Celsius. Show the formula and the answer, rounded to one decimal place."),
        ("p12_list_vs_tuple", "In one sentence, explain the difference between a list and a tuple in Python."),
    ]
    for pid, text in short:
        prompts.append({"id": pid, "text": text})

    return prompts


def load_prompts_file(path):
    data = json.loads(pathlib.Path(path).read_text())
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a JSON list of {{'id':..., 'text':...}} objects")
    return data


def load_jsonl(path):
    rows = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def do_compare(path_a, path_b):
    rows_a = {r["id"]: r for r in load_jsonl(path_a)}
    rows_b = {r["id"]: r for r in load_jsonl(path_b)}
    ids = [i for i in rows_a if i in rows_b]
    missing = (set(rows_a) | set(rows_b)) - set(ids)
    if missing:
        print(f"note: {len(missing)} prompt id(s) present in only one file, skipped: {sorted(missing)}")

    print(f"{'id':>24} {'match':>6} {'common':>8} {'first div':>10}  unit")
    print("-" * 62)
    identical = 0
    for pid in ids:
        ra, rb = rows_a[pid], rows_b[pid]
        ta, tb = ra.get("tokens") or [], rb.get("tokens") or []
        if ta and tb:
            seq_a, seq_b, unit = ta, tb, "tok"
        else:
            seq_a, seq_b, unit = list(ra.get("completion_text", "")), list(rb.get("completion_text", "")), "char"
        common = 0
        for x, y in zip(seq_a, seq_b):
            if x != y:
                break
            common += 1
        match = (seq_a == seq_b)
        first_div = None if match else common
        if match:
            identical += 1
        print(f"{pid:>24} {str(match):>6} {common:>8} {str(first_div):>10}  {unit}")

    print(f"\n{identical}/{len(ids)} identical")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_URL)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--label", default="run")
    ap.add_argument("--prompts-file", default=None,
                    help="JSON list of {'id':..., 'text':...}; default is the built-in 12-prompt set")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-busy-wait", action="store_true",
                    help="do not wait for the server to be idle (for runs under deliberate load)")
    ap.add_argument("--compare", nargs=2, metavar=("A_JSONL", "B_JSONL"), default=None,
                    help="skip run mode; pair rows by id and report where they diverge")
    args = ap.parse_args()

    if args.compare:
        do_compare(*args.compare)
        return

    key = API_KEY
    if args.prompts_file:
        prompts = load_prompts_file(args.prompts_file)
    else:
        prompts = default_prompts(args.base, args.model, key)

    out = RESULTS / f"lossless-{args.label}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(exist_ok=True)
    print(f"endpoint {args.base}   label {args.label}   {len(prompts)} prompts   "
          f"max_tokens {args.max_tokens}   temperature {args.temperature}   seed {args.seed}\n")
    print(f"{'id':>24} {'prompt tok':>11} {'out tok':>8} {'gen t/s':>9}  sha256")
    print("-" * 70)

    rows = []
    for p in prompts:
        while (not args.no_busy_wait) and busy(args.base):
            time.sleep(3)
        res = chat(args.base, args.model, key, p["text"], args.max_tokens,
                   args.temperature, args.seed)
        dt = max(res["wall"] - res["ttft"], 1e-9)
        gen_tok_s = round(res["ctok"] / dt, 2) if res["ctok"] else 0.0
        digest = hashlib.sha256(res["text"].encode()).hexdigest()
        row = {"id": p["id"], "label": args.label,
               "prompt_tokens": res["ptok"], "completion_tokens": res["ctok"],
               "ttft_s": round(res["ttft"], 3), "gen_tok_s": gen_tok_s,
               "completion_text": res["text"], "text_sha256": digest,
               "tokens": res["tokens"], "top2": res.get("top2", [])}
        rows.append(row)
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{p['id']:>24} {row['prompt_tokens']:>11} {row['completion_tokens']:>8} "
              f"{row['gen_tok_s']:>9.2f}  {digest[:16]}")

    print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
