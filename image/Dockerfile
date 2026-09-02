# DFlash2 overlay on the pinned Qwen3.8-27B day-0 SGLang image.
# Base: lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6
#        (sglang 0.0.0.dev0+qwen38.27b.g561c8f3, validated SM121 cell for this campaign)
# Overlay: sgl-project/sglang merge commit c14312a66420b75ca9a11bf1817c4db1fa26b097
#          (PR #35371 "DFlash2: local convolution + candidate selector", merged 2026-08-19)
#        + PR #35496 "Support quantized target lm_head in the DFlash2 selector"
#          (merged 2026-08-20). Without it the selector is refused the CUDA
#          graph on an NVFP4 target -- the log line is
#          "DFLASH draft greedy head kept eager (reason=quantized lm_head)".
# Scope: exactly the 6 runtime files + 2 unit tests the PR touched. Pure Python;
#        no CUDA build, no dependency changes, target-side code untouched.
FROM lmsysorg/sglang:qwen38-27b@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6
COPY python/sglang/kernels/ops/speculative/dflash.py /sgl-workspace/sglang/python/sglang/kernels/ops/speculative/dflash.py
COPY python/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py /sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py
COPY python/sglang/srt/models/dflash.py /sgl-workspace/sglang/python/sglang/srt/models/dflash.py
COPY python/sglang/srt/speculative/dflash2_compat.py /sgl-workspace/sglang/python/sglang/srt/speculative/dflash2_compat.py
COPY python/sglang/srt/speculative/dflash_utils.py /sgl-workspace/sglang/python/sglang/srt/speculative/dflash_utils.py
COPY python/sglang/srt/speculative/dflash_worker_v2.py /sgl-workspace/sglang/python/sglang/srt/speculative/dflash_worker_v2.py
COPY test/registered/unit/spec/test_dflash_logits.py /sgl-workspace/sglang/test/registered/unit/spec/test_dflash_logits.py
COPY test/registered/unit/model_executor/model_runner_components/test_spec_aux_hidden_state.py /sgl-workspace/sglang/test/registered/unit/model_executor/model_runner_components/test_spec_aux_hidden_state.py
LABEL org.opencontainers.image.title="qwen38-27b-sglang-dflash2-sm121" \
      org.opencontainers.image.description="Day-0 pinned image + DFlash2 PR 35371 overlay" \
      io.r0b0tlab.base-digest="sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6" \
      io.r0b0tlab.sglang-base="0.0.0.dev0+qwen38.27b.g561c8f3" \
      io.r0b0tlab.dflash2-commit="c14312a66420b75ca9a11bf1817c4db1fa26b097" \
      io.r0b0tlab.dflash2-pr="sgl-project/sglang#35371,#35496"
