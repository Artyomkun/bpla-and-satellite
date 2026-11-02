#!/usr/bin/env bash
#
# Instantiate the BPLADetection project on the riscv64 Julia.
# Single pass over the default pkg server. No mirrors/fallbacks.
#
#   bash instantiate_riscv64.sh /path/to/riscv64/julia /path/to/src
#
set -uo pipefail

JULIA_BIN=${1:?usage: instantiate_riscv64.sh <riscv64 julia> <project dir>}
PROJ=${2:?usage: instantiate_riscv64.sh <riscv64 julia> <project dir>}

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/.julia-riscv64}"
mkdir -p "$JULIA_DEPOT_PATH"

"$JULIA_BIN" --version
"$JULIA_BIN" --project="$PROJ" -e 'using Pkg; Pkg.instantiate()'
"$JULIA_BIN" --project="$PROJ" -e 'using Pkg; Pkg.precompile()' || {
  echo "precompile reported issues; rerun with JULIA_DEBUG=loading for detail" >&2
  exit 1
}
echo "OK"