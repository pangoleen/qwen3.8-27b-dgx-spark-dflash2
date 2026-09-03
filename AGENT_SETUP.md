# Agent setup prompt

Paste everything below this line into a coding agent that has shell access on
the DGX Spark. It assumes this repository is cloned in the current directory.

---

You are setting up the Qwen3.8-27B serving recipe in this repository on an
NVIDIA DGX Spark (GB10, 128 GB unified memory, aarch64). Work through the steps
in order, verify each one before moving on, and report what you found at every
step. Rules: never print, log, or paste the contents of the API key file; never
run two large model servers at once; every operation on the tactic cache goes
through a container, never a host-side copy; if a step fails, stop and report
rather than improvising around it.

1. Check the box: `nvidia-smi` or `nvidia-ctk --version`, `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi` (or the base image you already have) to confirm the GPU is reachable from Docker; `df -h $HOME` for at least 80 GB free; `free -g` and `docker ps` to confirm no other large model server is running.
2. Read README.md fully. Then read serve.sh and .env.sample.
3. Weights: follow the README "Weights" section exactly, with the two pinned revisions, into `$HOME/models/hf`. A `--revision <sha>` download writes no `refs/` directory at all; run the README loop that creates it (`serve.sh` writes it too, at launch). Confirm both snapshot directories exist and contain `.safetensors` files, and that `refs/main` in each names a snapshot that holds them.
4. API key: create `$HOME/models/vllm_api_key.txt` with `openssl rand -hex 32`, mode 600. Do not display it.
5. Engine image: `docker build -t qwen38-27b-sglang-dflash2-sm121:0.3.0 -f image/Dockerfile image/`. Report the resulting image ID.
6. Tactic cache: run the container command from the README's tactic-cache section to install both draws into `$HOME/sglang-cache`. Confirm with the `ls` it prints that both `rank_tp0_pp0_dp0.json` files exist and are mode 600.
7. Launch: `cp .env.sample .env`, `source .env`, `./serve.sh`. Wait for it to print `ready after`, then `tokenize=200` and `restarts=0`. Boot takes 170-240 s. Then check the log: `docker logs sglang38-dflash2 2>&1 | grep -c "kept eager"` must be 0, and report the `max_total_num_tokens` value (expected about 413,460).
8. First token: run the quickstart curl from the README. Report the completion and the usage block.
9. One rung of the sweep: `python3 bench/ctxsweep.py --label agent-check --lengths 512,8192 --reps 2 --out-tokens 512`. Report the table. Expected generation is roughly 64-78 tok/s at both rungs with accepted tokens per pass around 7-8; if it is under 55, report that the tactic draw did not take (check the cache mount and the `ls` from step 6) rather than re-tuning.
10. Finish with a summary: image ID, boot time, KV pool size, the two sweep rows, and anything that differed from what the README said to expect.
