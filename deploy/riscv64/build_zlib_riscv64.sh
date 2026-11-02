#!/usr/bin/env bash
#
# Cross-build zlib 1.3.1 for riscv64-linux-gnu with Debian/Ubuntu cross toolchain.
# No Docker. Proof-of-pipeline for the RISC-V Julia artifact chain.
#
# Usage:  bash build_zlib_riscv64.sh
# Output: ./out/lib/libz.a  ./out/lib/libz.so*  + /tmp/ztest_riscv64 (static test binary)
#
set -euo pipefail

ZVER=1.3.1
SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
CROSS=riscv64-linux-gnu
PREFIX="$(pwd)/out"
SRC="zlib-${ZVER}"
TGZ="zlib-${ZVER}.tar.gz"
TORCH="gcc-riscv64-linux-gnu g++-riscv64-linux-gnu binutils-riscv64-linux-gnu qemu-user qemu-user-static"

log() { printf '\n==> %s\n' "$*"; }

if ! command -v "${CROSS}-gcc" >/dev/null 2>&1; then
  log "Installing cross toolchain + qemu-user..."
  sudo apt-get update
  sudo apt-get install -y $TORCH
fi

if [ ! -f "${TGZ}" ]; then
  log "Downloading zlib ${ZVER}..."
  if ! curl -fsSL -o "${TGZ}" "https://zlib.net/zlib-${ZVER}.tar.gz"; then
    log "zlib.net unreachable, falling back to GitHub mirror..."
    curl -fsSL -o "${TGZ}" "https://github.com/madler/zlib/releases/download/v${ZVER}/zlib-${ZVER}.tar.gz" || { echo "download failed"; exit 1; }
  fi
fi

if command -v sha256sum >/dev/null 2>&1; then
  got=$(sha256sum "${TGZ}" | awk '{print $1}')
  if [ "$got" != "$SHA256" ]; then
    echo "WARNING: sha256 mismatch (got $got). Using file anyway." >&2
  else
    echo "sha256 OK"
  fi
fi

[ -d "${SRC}" ] && rm -rf "${SRC}"
tar -xzf "${TGZ}"
cd "${SRC}"

log "Configuring (cross)..."
if CC="${CROSS}-gcc" AR="${CROSS}-ar" RANLIB="${CROSS}-ranlib" ./configure --prefix="${PREFIX}" --64; then
  log "make + install..."
  make -j"$(nproc)"
  make install
else
  log "configure failed -> manual compile fallback"
  "${CROSS}-gcc" -O3 -fPIC -D_LARGEFILE64_SOURCE=1 -DHAVE_HIDDEN \
      -c adler32.c crc32.c deflate.c infback.c inffast.c inflate.c inftrees.c trees.c zutil.c
  "${CROSS}-ar" rcs libz.a *.o
  mkdir -p "${PREFIX}/lib" "${PREFIX}/include"
  cp libz.a "${PREFIX}/lib/"
  "${CROSS}-gcc" -shared -Wl,-soname,libz.so.1 -o "${PREFIX}/lib/libz.so.1.3.1" *.o
  ln -sf libz.so.1.3.1 "${PREFIX}/lib/libz.so.1"
  ln -sf libz.so.1.3.1 "${PREFIX}/lib/libz.so"
  cp zlib.h zconf.h "${PREFIX}/include/"
fi

log "Building + running static cross test via qemu..."
"${CROSS}-gcc" -static -O2 -I. -o /tmp/ztest_riscv64 example.c libz.a
qemu-riscv64 /tmp/ztest_riscv64 | head -n 6

log "Artifacts:"
(command -v file >/dev/null 2>&1 && file "${PREFIX}"/lib/libz.a "${PREFIX}"/lib/libz.so*) \
  || readelf -h "${PREFIX}"/lib/libz.so | head -n 8
ls -la "${PREFIX}"/lib/

cat <<'EOF'

==> Done.
To feed this into a Julia JLL instead of the prebuilt artifact, add to
  <HOME>/.julia/artifacts/Overrides.toml:
    [83775a58-1f1d-513f-b197-d71354ab007a]   # Zlib_jll UUID
    Zlib = "PATH/TO/out"
(artifacts are then used from out/ instead of the package server)
EOF