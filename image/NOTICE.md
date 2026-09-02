# Engine image overlay

`Dockerfile` builds `qwen38-27b-sglang-dflash2-sm121:0.3.0`: the pinned day-0
Qwen3.8-27B SGLang image (`lmsysorg/sglang:qwen38-27b`, digest in the Dockerfile)
plus the files touched by two upstream SGLang pull requests, copied verbatim
from `sgl-project/sglang`:

- #35371 "DFlash2: local convolution + candidate selector" (merge c14312a)
- #35496 "Support quantized target lm_head in the DFlash2 selector"

Six runtime files and two unit tests, pure Python, no CUDA build. They are
SGLang code under the Apache License 2.0; copyright the SGLang contributors.
The overlay approach and label scheme follow r0b0tlab's community image.

```bash
docker build -t qwen38-27b-sglang-dflash2-sm121:0.3.0 -f image/Dockerfile image/
```

After boot, `serve.sh` greps the log for `kept eager (reason=quantized lm_head)`;
if that line appears, #35496 is missing and the selector runs outside the draft
CUDA graph.
