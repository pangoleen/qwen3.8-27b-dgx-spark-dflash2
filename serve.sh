#!/usr/bin/env bash
# serve.sh — SGLang + DFlash2 speculative decoding for Qwen3.8-27B on one
# DGX Spark (GB10), port 8003. Every knob is overridable from the environment,
# so a sweep can vary one variable without editing this file:
#   DRAFT_TOKENS=24 ./serve.sh
set -euo pipefail

# 0.3.0 carries upstream PR #35496, which folds the DFlash2 selector into the
# draft CUDA graph on the NVFP4 packed lm_head. Older builds run it eager.
IMAGE=${IMAGE:-qwen38-27b-sglang-dflash2-sm121:0.3.0}
NAME=${NAME:-sglang38-dflash2}
PORT=${PORT:-8003}
MODEL=${MODEL:-RadixArk/Qwen3.8-27B-NVFP4}

# The NVFP4 drafter: 1.45 GB of weights per pass instead of 3.85, acceptance
# unchanged, +5 to +10% generation. It needs its scheme named explicitly.
DRAFT=${DRAFT:-maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal}
# Checkpoint snapshots these numbers were measured on. Empty = whatever refs/main
# points at in your cache (see README, Weights).
REVISION=${REVISION:-}
DRAFT_REVISION=${DRAFT_REVISION:-}
DRAFT_QUANT=${DRAFT_QUANT:-modelopt_fp4}

# 16, not the 8 the DFlash2 blog uses. The drafter's block_size of 8 is not a
# cap on the verify budget: 8 -> 16 is +69.3% edit-heavy and +10.5% fresh.
# 24 is past the knee; both workloads fall.
DRAFT_TOKENS=${DRAFT_TOKENS:-16}
DRAFT_ATTN=${DRAFT_ATTN:-flashinfer}

MEM_FRACTION=${MEM_FRACTION:-0.65}
CHUNK=${CHUNK:-2048}
KV_DTYPE=${KV_DTYPE:-bfloat16}
SSM_DTYPE=${SSM_DTYPE:-bfloat16}
CTX=${CTX:-262144}

# 4 gives a 413,460-token KV pool, room for the full 262,144 context plus
# more. 16 allows sixteen concurrent requests and shrinks the pool to
# 238,605 tokens, below the context window.
MAX_RUNNING=${MAX_RUNNING:-4}

# AUTOTUNE=1 keeps FlashInfer autotuning on, which is what lets the mounted
# tactic cache be replayed. With it off the good draw is ignored.
AUTOTUNE=${AUTOTUNE:-1}
SGLANG_CACHE=${SGLANG_CACHE:-$HOME/sglang-cache}
HF_CACHE=${HF_CACHE:-$HOME/models/hf}

# Pinning to the big cores. Inherited from a community recipe and never
# measured here. Set CPUSET="" to run unpinned.
CPUSET=${CPUSET-5-9,15-19}

# SPEC=0 launches the same target with no drafter at all (the losslessness
# baseline).
SPEC=${SPEC:-1}

# 1 disables prefill CUDA graphs. Inherited from the community stack; under
# DFlash2 the drafter does real work during prefill, so a graph may now pay.
DISABLE_PREFILL_GRAPH=${DISABLE_PREFILL_GRAPH:-1}

# The drafter's own config declares a 2048-token sliding window and this build
# applies it, so the flag below only compacts draft KV storage. Kept for
# reference and refused: it parses and boots, then kills the scheduler.
DRAFT_WINDOW=${DRAFT_WINDOW:-}
if [ -n "$DRAFT_WINDOW" ]; then
  echo "REFUSED: DRAFT_WINDOW is broken on this image (see README, Traps)." >&2
  echo "The overlay worker dies on the first batch with" >&2
  echo "  AttributeError: 'DFlashDraftInputV2' object has no attribute 'nxt_kv_lens_cpu'" >&2
  exit 2
fi

# The key is read from a file, never passed on a command line or baked in.
SPARK_API_KEY_FILE=${SPARK_API_KEY_FILE:-$HOME/models/vllm_api_key.txt}
[ -r "$SPARK_API_KEY_FILE" ] || { echo "no API key file at $SPARK_API_KEY_FILE (see README, Quickstart)" >&2; exit 2; }
KEY=$(cat "$SPARK_API_KEY_FILE")
[ -d "$HF_CACHE/hub" ] || { echo "HF_CACHE=$HF_CACHE has no hub/ directory; download the checkpoints first (README, Weights)" >&2; exit 2; }

# HF cache refs. `hf download --revision <sha>` writes no refs/ directory at
# all, and one call inside SGLang reads the DRAFT config with no revision: the
# speculative-algorithm alias resolver in arg_groups/speculative_hook.py, which
# runs before any DFLASH handling. Offline that lookup goes through refs/main,
# so a clean SHA-only follow of the README crash-loops before the engine ever
# reads --speculative-draft-model-revision. We know both SHAs, so write the
# refs here. Point refs/main only at a snapshot that holds weights: a partial
# download can leave it naming a tokenizer-only snapshot (README, Weights).
write_ref() {                                   # write_ref <repo-id> <sha>
  local dir="$HF_CACHE/hub/models--${1//\//--}"
  if [ -z "${2:-}" ]; then
    # No pin (the reader skipped `source .env`, or uses their own cache).
    # Say so now rather than after a 200 s boot that ends in a crash loop.
    if [ -d "$dir" ] && [ ! -f "$dir/refs/main" ]; then
      echo "warning: $1 has no refs/main and no revision pinned; the offline boot will fail (README, Weights)" >&2
    fi
    return 0
  fi
  if [ ! -d "$dir/snapshots/$2" ]; then
    echo "warning: $1 has no snapshot $2 in $HF_CACHE; refs/main not written (README, Weights)" >&2
    return 0
  fi
  if ! compgen -G "$dir/snapshots/$2/*.safetensors" >/dev/null; then
    echo "warning: $1 snapshot $2 holds no .safetensors; refs/main left alone (README, Weights)" >&2
    return 0
  fi
  [ "$(cat "$dir/refs/main" 2>/dev/null)" = "$2" ] && return 0
  if mkdir -p "$dir/refs" 2>/dev/null && printf '%s' "$2" > "$dir/refs/main" 2>/dev/null; then
    echo "refs/main    $1 -> $2"
  else
    echo "warning: cannot write $dir/refs/main (root-owned cache?); see README, Weights" >&2
  fi
  return 0
}
write_ref "$MODEL" "$REVISION"
if [ "$SPEC" = "1" ]; then write_ref "$DRAFT" "$DRAFT_REVISION"; fi

# Refuse to boot without a usable refs/main. Offline, the alias resolver above
# reads it, so a missing one costs a 200 s boot that ends in a crash loop, and
# `HF_HUB_OFFLINE=0` then looks like the fix when the real fault is one file.
check_ref() {                                   # check_ref <repo-id>
  local dir="$HF_CACHE/hub/models--${1//\//--}" sha
  sha=$(cat "$dir/refs/main" 2>/dev/null) || true
  if [ -z "$sha" ]; then
    echo "$1 has no $dir/refs/main. The offline boot needs it. Write it with:" >&2
    echo "  mkdir -p $dir/refs && printf '%s' <snapshot-sha> > $dir/refs/main" >&2
    echo "See README, Weights for both SHAs; \`source .env\` and serve.sh writes them for you." >&2
    exit 2
  fi
  if [ ! -d "$dir/snapshots/$sha" ]; then
    echo "$1 refs/main names $sha, which is not in $dir/snapshots (README, Weights)" >&2
    exit 2
  fi
  if ! compgen -G "$dir/snapshots/$sha/*.safetensors" >/dev/null; then
    echo "$1 refs/main names $sha, which holds no .safetensors - a partial download (README, Weights)" >&2
    exit 2
  fi
}
check_ref "$MODEL"
if [ "$SPEC" = "1" ]; then check_ref "$DRAFT"; fi

mkdir -p "$SGLANG_CACHE"

docker rm -f "$NAME" >/dev/null 2>&1 || true

EXTRA="${EXTRA_ARGS:-}"
[ "$AUTOTUNE" = "0" ] && case "$EXTRA" in *disable-flashinfer-autotune*) ;; *) EXTRA="$EXTRA --disable-flashinfer-autotune" ;; esac
[ "$DISABLE_PREFILL_GRAPH" = "1" ] && EXTRA="$EXTRA --disable-prefill-cuda-graph"

if [ "$SPEC" = "1" ]; then
  SPEC_ARGS="--speculative-algorithm DFLASH --speculative-draft-model-path $DRAFT ${DRAFT_REVISION:+--speculative-draft-model-revision $DRAFT_REVISION} --speculative-num-draft-tokens ${DRAFT_TOKENS} --speculative-draft-attention-backend ${DRAFT_ATTN} ${DRAFT_QUANT:+--speculative-draft-model-quantization $DRAFT_QUANT}"
else
  SPEC_ARGS=""
fi
echo "spec         $SPEC   draft quant ${DRAFT_QUANT:-checkpoint default}"
echo "image        $IMAGE"
echo "drafter      $DRAFT  (tokens $DRAFT_TOKENS, attn $DRAFT_ATTN)"
echo "mem-fraction $MEM_FRACTION   chunk $CHUNK   kv $KV_DTYPE   ssm $SSM_DTYPE"
echo "max-running  $MAX_RUNNING   autotune $AUTOTUNE   tactic cache $SGLANG_CACHE"

# --restart unless-stopped survives a host reboot and still respects a
# deliberate docker stop. It also hides a crash loop: check RestartCount.
docker run -d --name "$NAME" --restart unless-stopped \
  --gpus all --shm-size 32g --ipc=host \
  ${CPUSET:+--cpuset-cpus $CPUSET} -p ${PORT}:${PORT} \
  -v "${HF_CACHE}":/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -v "${SGLANG_CACHE}":/root/.cache/sglang \
  "$IMAGE" \
  sglang serve --model-path "$MODEL" ${REVISION:+--revision $REVISION} --served-model-name qwen3.8-27b \
  --trust-remote-code --host 0.0.0.0 --port ${PORT} --api-key "$KEY" \
  --context-length ${CTX} --mem-fraction-static ${MEM_FRACTION} \
  --attention-backend flashinfer --kv-cache-dtype ${KV_DTYPE} \
  --chunked-prefill-size ${CHUNK} --mamba-ssm-dtype ${SSM_DTYPE} \
  --mamba-radix-cache-strategy extra_buffer --page-size 1 \
  --mamba-full-memory-ratio 11.93 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs "{\"enable_thinking\": false, \"preserve_thinking\": false}" \
  --max-running-requests ${MAX_RUNNING} --enable-metrics ${EXTRA} \
  $SPEC_ARGS

echo
echo "waiting for health on :${PORT} ..."
for i in $(seq 1 120); do
  if curl -sf -H "Authorization: Bearer $KEY" \
       http://127.0.0.1:${PORT}/v1/models >/dev/null 2>&1; then
    echo "ready after ${i}0s"
    # A config bug that hides behind a healthy-looking server is the failure
    # mode this project keeps rediscovering. Make this one visible.
    if docker logs "$NAME" 2>&1 | grep -q "kept eager (reason=quantized lm_head)"; then
      echo
      echo "WARNING: the DFlash2 selector is running EAGER, outside the draft"
      echo "CUDA graph. This image predates upstream PR #35496. Expect lower"
      echo "throughput. Use IMAGE=qwen38-27b-sglang-dflash2-sm121:0.3.0"
    else
      echo "selector: folded into the draft cuda graph (PR #35496 present)"
    fi
    # /v1/models answers before the scheduler is live. /tokenize does not.
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
      -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" \
      -d '{"model":"qwen3.8-27b","prompt":"hi"}' \
      http://127.0.0.1:${PORT}/tokenize)
    echo "tokenize: $code   restarts: $(docker inspect "$NAME" --format '{{.RestartCount}}')"
    [ "$code" = "200" ] || { echo "scheduler is not serving; check docker logs $NAME" >&2; exit 1; }
    exit 0
  fi
  sleep 10
done
echo "TIMEOUT — check: docker logs $NAME"
exit 1
