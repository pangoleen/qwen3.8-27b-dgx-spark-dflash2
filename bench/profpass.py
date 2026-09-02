#!/usr/bin/env python3
"""
profpass — break an SGLang torch-profiler trace into per-pass GPU time.

Answers "where does a speculative decode pass spend its 113 ms": GPU busy vs
idle, gaps between kernels, and the top kernels per pass. Written for the
DFlash2 server; works on any SGLang trace that carries the `scheduler.run_batch`
and `step[EXTEND ...]` user annotations.

Capture on the box. No restart is needed; the profiler directory defaults to
/tmp inside the container.

    export SPARK_API_KEY=$(cat ~/models/vllm_api_key.txt)
    B=${SPARK_BASE_URL:-http://localhost:8003/v1}; B=${B%/v1}
    curl -s $B/start_profile -H "Authorization: Bearer $SPARK_API_KEY" \
      -H 'Content-Type: application/json' \
      -d '{"output_dir":"/tmp/prof","activities":["CPU","GPU"]}'
    ... send one chat request ...
    curl -s $B/stop_profile -H "Authorization: Bearer $SPARK_API_KEY"
    docker cp sglang38-dflash2:/tmp/prof ~/prof

Usage:
    python3 bench/profpass.py <trace.json.gz> [--top 30]

Method: the decode window starts when the EXTEND (prefill) step's CPU span
ends and runs to the last kernel. Passes = `scheduler.run_batch` calls in that
window. GPU busy = union of kernel/memcpy/memset intervals, so overlapping
streams are not double counted.
"""
import argparse, collections, gzip, json, statistics as st


def union(iv):
    tot = 0; cur = None
    for s, e in sorted(iv):
        if cur is None or s > cur[1]:
            if cur: tot += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur: tot += cur[1] - cur[0]
    return tot


def group(k):
    nm = k['name'].lower()
    if k.get('cat') == 'gpu_memcpy': return 'memcpy'
    if 'cutlass_80' in nm: return 'gemm bf16 (drafter, sm80 wmma)'
    if any(s in nm for s in ('fp4', 'nvfp4', 'gemmuniversal')): return 'gemm fp4 (mlp)'
    if 'nvjet' in nm or 'cublas' in nm: return 'gemm fp8 (attn/gdn proj)'
    if any(s in nm for s in ('attention', 'flashinfer::batch', 'paged', 'mergestates')): return 'attention'
    if any(s in nm for s in ('gated', 'delta', 'recurrent', 'causal_conv', 'mamba', 'qkvzba')): return 'gdn'
    if any(s in nm for s in ('norm', 'elementwise', 'silu', 'rotary', 'rope', 'act_and_mul', 'quant')): return 'norm/elementwise/quant'
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--top', type=int, default=30)
    a = ap.parse_args()
    opener = gzip.open if a.trace.endswith('.gz') else open
    ev = json.load(opener(a.trace))['traceEvents']
    X = [e for e in ev if e.get('ph') == 'X']
    rb = sorted([e for e in X if e.get('cat') == 'user_annotation' and e['name'] == 'scheduler.run_batch'], key=lambda e: e['ts'])
    ext = [e for e in X if e.get('cat') == 'user_annotation' and e['name'].startswith('step[EXTEND')]
    kern = sorted([e for e in X if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')], key=lambda e: e['ts'])
    t0 = (ext[0]['ts'] + ext[0]['dur']) if ext else rb[0]['ts']
    t1 = max(k['ts'] + k['dur'] for k in kern)
    dk = [k for k in kern if k['ts'] >= t0]
    n = len([e for e in rb if e['ts'] >= t0]) or 1
    busy = union([(k['ts'], k['ts'] + k['dur']) for k in dk])
    print(f"decode window {(t1-t0)/1000:.1f} ms, passes {n}, kernels {len(dk)}")
    print(f"per pass: wall {(t1-t0)/n/1000:.1f} ms | gpu busy {busy/n/1000:.1f} ms | gpu idle {(t1-t0-busy)/n/1000:.1f} ms | kernels {len(dk)/n:.0f}")
    iv = sorted((k['ts'], k['ts'] + k['dur']) for k in dk); gaps = []; cur = iv[0][1]
    for s, e in iv[1:]:
        if s > cur: gaps.append(s - cur)
        cur = max(cur, e)
    big = [g for g in gaps if g > 200]
    print(f"gpu gaps >0.2 ms: {len(big)} totalling {sum(big)/1000:.1f} ms ({sum(big)/n/1000:.2f} ms/pass), largest {max(gaps)/1000:.1f} ms")
    w = [e['dur'] / 1000 for e in rb if e['ts'] >= t0]
    print(f"run_batch cpu wall per pass: median {st.median(w):.1f} ms (min {min(w):.1f}, max {max(w):.1f})")
    grp = collections.Counter()
    for k in dk: grp[group(k)] += k['dur']
    print("\nper pass by group (ms):")
    for g, us in grp.most_common(): print(f"  {us/n/1000:6.1f}  {g}")
    agg = collections.Counter(); cnt = collections.Counter()
    for k in dk: agg[k['name'][:100]] += k['dur']; cnt[k['name'][:100]] += 1
    print(f"\ntop kernels per pass (ms, x calls):")
    for name, us in agg.most_common(a.top): print(f"  {us/n/1000:6.2f}  x{cnt[name]/n:6.1f}  {name}")


if __name__ == '__main__':
    main()
