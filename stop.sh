#!/usr/bin/env bash
# Stop the server. The container has --restart unless-stopped, so `docker stop`
# is the only thing that keeps it down across a reboot.
docker stop "${NAME:-sglang38-dflash2}"
