#!/usr/bin/env bash
#
# Run INSIDE the riscv64 guest: natively build Julia and instantiate the project.
# Copy the source tree into the guest first (scp or virtio-9p).
#
#   bash vm_provision_julia.sh [JULIA_VER=v1.12.7] [PROJECT_DIR]
#
set -euo pipefail

JVER=${1:-v1.12.7}
PROJ=${2:-}
JOBS=${JOBS:-2}          # keep small under TCG emulation
export JULIA_NUM_THREADS=1

log() { printf '\n==> %s\n' "$*"; }

log "installing Julia build deps"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential cmake curl git gfortran pkg-config python3 ca-certificates

log "fetching Julia ${JVER}"
[ -d julia ] || git clone --depth 1 --branch "${JVER}" https://github.com/JuliaLang/julia.git
cd julia
[ -f Make.user ] || touch Make.user

log "native build (JOBS=${JOBS}) — long, hours under emulation"
make -j"${JOBS}"

log "verify"
file usr/bin/julia
./julia --version

if [ -n "${PROJ}" ]; then
  log "instantiate ${PROJ}"
  ./julia --project="${PROJ}" -e 'using Pkg; Pkg.instantiate()'
  ./julia --project="${PROJ}" -e 'using Pkg; Pkg.precompile()'
  log "load smoke test"
  ./julia --project="${PROJ}" -e 'using BPLADetection; using Flux; using Images; @info "BPLADetection loads OK on riscv64"'
fi

cat <<EOF

Julia built: $(pwd)/usr/bin/julia
Result log, if instantiate available: above.
EOF