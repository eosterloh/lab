#!/usr/bin/env bash
# Cloud Agent install for the `lab` harness.
#
# Installs into the interpreter's GLOBAL site-packages (via sudo) on purpose:
# the `lab` trainer runs as a sandboxed subprocess that strips the environment
# and sets HOME to the job dir, so a per-user (`pip install --user`) install of
# torch would be invisible to it. A global install lives on the interpreter's
# default sys.path regardless of HOME, so the sandboxed trainer can import torch.
#
# Idempotent: safe to run repeatedly. Uses the repo's pinned extras and a
# CPU-only torch wheel (Cloud Agent VMs have no GPU).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

py=python3

# Prefer sudo for the global site-packages; fall back to a plain invocation if
# this user can already write there (e.g. running as root).
if sudo -n true 2>/dev/null; then
  pip_run=(sudo -n "$py" -m pip)
else
  pip_run=("$py" -m pip)
fi

# CPU-only torch (enables the `lab` trainer + its tests; no CUDA on Cloud VMs).
"${pip_run[@]}" install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu

# The package itself plus dev (pytest) and data (datasets/huggingface_hub) extras.
# Editable so source edits are picked up without reinstalling.
"${pip_run[@]}" install -e ".[dev,data]"

echo "cloud-agent-install: done"
