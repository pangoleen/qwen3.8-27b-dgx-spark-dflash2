#!/usr/bin/env python3
"""
accfix — fixed-output acceptance test across the context ladder.

ctxsweep measures accepted tokens per pass as depth grows, but the task it
appends is free-form, so a change could be the drafter losing acceptance OR the
model writing a different kind of answer at each depth. This test removes that
confound: at every prefix length the model reproduces one fixed ~70-line Python
class, character for character. The wanted output is identical at every depth,
so any change in tokens per pass isolates the drafter's conditioning.

Helpers and the seed corpus come from ctxsweep.py in this folder, so both tools
build the same prompt at the same rung.

Environment:
    SPARK_BASE_URL   default http://localhost:8003/v1
    SPARK_API_KEY    the server's API key
    SPARK_MODEL      default qwen3.8-27b

Usage:
    export SPARK_API_KEY=$(cat ~/models/vllm_api_key.txt)
    python3 bench/accfix.py --label depth
    python3 bench/accfix.py --lengths 1024,65536 --reps 2   # quick check
"""
import argparse, json, os, pathlib, re, statistics, time, urllib.request, uuid
from ctxsweep import (API_KEY, BASE_URL, MODEL, RESULTS, _hdr, build_prompt,
                      busy, draft_budget, gauge, seed_corpus)

HERE = pathlib.Path(__file__).parent
CTX_LIMIT = 262144  # the server's --context-length

# A fixed reference class. The content does not matter beyond being long enough
# to need many decode passes and unambiguous to copy.
REFERENCE_FUNC = '''class LRUCache:
    """Fixed-capacity cache with O(1) get/put, evicting the least recently
    used entry when full. Backed by a dict plus an intrusive doubly linked
    list so no entry ever needs a linear scan."""

    class _Node:
        __slots__ = ("key", "value", "prev", "next")

        def __init__(self, key=None, value=None):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.map = {}
        self.head = self._Node()
        self.tail = self._Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _push_front(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        node = self.map.get(key)
        if node is None:
            return -1
        self._remove(node)
        self._push_front(node)
        return node.value

    def put(self, key, value):
        node = self.map.get(key)
        if node is not None:
            node.value = value
            self._remove(node)
            self._push_front(node)
            return
        if len(self.map) >= self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.map[lru.key]
        node = self._Node(key, value)
        self.map[key] = node
        self._push_front(node)

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def clear(self):
        self.map.clear()
        self.head.next = self.tail
        self.tail.prev = self.head
'''

TASK = ("\n\n---\nReproduce the following Python function exactly, character "
        "for character, in one code block and nothing else:\n\n```python\n"
        + REFERENCE_FUNC + "```\n")


def chat_text(base, model, key, prompt, max_tokens, timeout=3600):
    """Like ctxsweep.chat(), but keeps the completion text for the match check."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
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
            "text": "".join(chunks)}


def extract_code(text):
    """Body of the first fenced code block, or the stripped raw text."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip("\n")


def first_diff_line(got, want):
    """The first line where got differs from want, or None if they match."""
    if got.strip("\n") == want.strip("\n"):
        return None
    g_lines, w_lines = got.split("\n"), want.split("\n")
    for i, (g, w) in enumerate(zip(g_lines, w_lines), 1):
        if g != w:
            return f"line {i}: got {g!r} want {w!r}"
    if len(g_lines) != len(w_lines):
        return f"length mismatch: got {len(g_lines)} lines, want {len(w_lines)} lines"
    return "differs but no line-level diff found"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_URL)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--label", default="accfix")
    ap.add_argument("--lengths", default="1024,65536,262144")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=600)
    args = ap.parse_args()

    key = API_KEY
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    budget = draft_budget(args.base, key)
    reference = REFERENCE_FUNC.strip("\n")
    seed = seed_corpus()

    out = RESULTS / f"accfix-{args.label}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(exist_ok=True)
    print(f"endpoint {args.base}   draft budget {budget}   max_tokens {args.max_tokens}   reps {args.reps}\n")
    print(f"{'ctx':>8} {'prompt tok':>11} {'gen t/s':>9} {'tok/pass':>9} {'sat':>5} {'out tok':>8}  match")
    print("-" * 74)

    # The context limit covers prompt plus output; a larger target is refused
    # with HTTP 400 at the request.
    lengths = [min(L, CTX_LIMIT - args.max_tokens - 512) for L in lengths]

    rows = []
    for L in lengths:
        while busy(args.base):
            time.sleep(3)
        tag = uuid.uuid4().hex[:8]
        try:
            prompt, ptok = build_prompt(args.base, args.model, key, L, tag, seed)
            prompt = prompt + TASK
        except Exception as e:
            print(f"{L:>8}  prompt build failed: {e}")
            continue

        gens, matches, diffs, out_toks = [], [], [], []
        for r in range(args.reps):
            res = chat_text(args.base, args.model, key, prompt, args.max_tokens)
            if res["ctok"] < 8:
                print(f"{L:>8}  only {res['ctok']} completion tokens on rep {r}; skipping rep")
                continue
            dt = max(res["wall"] - res["ttft"], 1e-9)
            gens.append(res["ctok"] / dt)
            out_toks.append(res["ctok"])
            diff = first_diff_line(extract_code(res["text"]), reference)
            matches.append(diff is None)
            diffs.append(diff)
        if not gens:
            print(f"{L:>8}  no usable reps; skipping")
            continue

        tpp = gauge(args.base, "sglang:spec_accept_length")
        sat = (tpp / budget) if (tpp and budget) else None
        matched = all(matches)
        first_fail = next((d for d in diffs if d is not None), None)
        row = {"label": args.label, "ctx_target": L, "prompt_tokens": ptok,
               "gen_tok_s": round(statistics.median(gens), 2),
               "tok_per_pass": tpp, "saturation": round(sat, 3) if sat else None,
               "out_tokens": out_toks[0] if out_toks else None,
               "reps": len(gens), "matched": matched,
               "rep_matches": matches, "first_diff_line": first_fail}
        rows.append(row)
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{L:>8} {ptok:>11} {row['gen_tok_s']:>9.2f} {str(tpp):>9} "
              f"{(f'{100*sat:.0f}%' if sat else '-'):>5} "
              f"{str(row['out_tokens']):>8}  {'OK' if matched else 'MISMATCH: ' + (first_fail or '')}")

    print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
