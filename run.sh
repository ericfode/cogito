#!/usr/bin/env bash
# Cogito runner — sets up NixOS/WSL2 CUDA paths and env
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# CUDA setup for WSL2 + NixOS
export PATH="${SCRIPT_DIR}/.local/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/nix/store/5f722gfp28da7xdnak1pv6gjjdffz2ry-cuda12.8-cuda_cudart-12.8.90/lib:/nix/store/vl8jkqpr0l3fac3cxiy4nwc5paiww1lv-zlib-1.3.1/lib:${LD_LIBRARY_PATH:-}"
export CPATH="/nix/store/6wqqrp57pi7ziy3wxpsxdmkfl481b43v-cuda_cudart-12.8.90-dev/include:${CPATH:-}"
export CC="/nix/store/41zf34cqqaqxcd6q4706vpvd51gwf4gf-clang-wrapper-19.1.7/bin/clang"

# Use CUDA by default (RTX 4090 via WSL2)
export CUDA=1

source "${SCRIPT_DIR}/.venv/bin/activate"
[ -f ~/.config/wandb_env ] && source ~/.config/wandb_env

exec python3 -m cogito.main "$@"
