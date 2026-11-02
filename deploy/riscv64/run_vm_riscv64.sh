#!/usr/bin/env bash
#
# Boot the riscv64 QEMU VM created by make_vm_riscv64.sh.
# Optional: expose a host directory into the guest via 9p:
#   MOUNT_DIR=/path/to/src bash run_vm_riscv64.sh [WORK]
#   (in guest: mkdir -p /mnt/host && mount -t 9p -o trans=virtio host0 /mnt/host)
#
set -euo pipefail

WORK=${1:-"$(pwd)/riscv64-vm"}
RAM=${RAM:-8G}
SMP=${SMP:-4}

ARGS=(
  -machine virt
  -m "${RAM}"
  -smp "${SMP}"
  -kernel "${WORK}/boot/vmlinuz"
  -initrd "${WORK}/boot/initrd"
  -append "root=/dev/vda rw console=ttyS0"
  -drive file="${WORK}/rootdisk.img,format=raw,if=virtio"
  -netdev user,id=n0
  -device virtio-net-pci,netdev=n0
  -nographic
)

if [ -n "${MOUNT_DIR:-}" ]; then
  ARGS+=( -virtfs local,path="${MOUNT_DIR}",mount_tag=host0,security_model=none,id=host0 )
  echo "host dir '${MOUNT_DIR}' available in guest at /mnt/host"
fi

exec qemu-system-riscv64 "${ARGS[@]}"