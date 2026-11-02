#!/usr/bin/env bash
#
# Create a riscv64 QEMU VM (Debian sid-ports) on the Ubuntu host.
# Produces: <WORK>/boot/vmlinuz, <WORK>/boot/initrd, <WORK>/rootdisk.img
#
#   bash make_vm_riscv64.sh [WORK]
#
set -euo pipefail

WORK=${1:-"$(pwd)/riscv64-vm"}
MIRROR="http://deb.debian.org/debian-ports"
DISK_SIZE=${DISK_SIZE:-32G}
RAM=${RAM:-8G}
SMP=${SMP:-4}
ROOTFS="${WORK}/rootfs"
BOOT="${WORK}/boot"

log() { printf '\n==> %s\n' "$*"; }

# ---------- 1/4 host tools ----------
log "1/4 host tools"
missing=()
command -v qemu-system-riscv64 >/dev/null || missing+=(qemu-system-misc)
command -v debootstrap       >/dev/null || missing+=(debootstrap)
command -v qemu-riscv64-static >/dev/null || missing+=(qemu-user-static)
command -v mkfs.ext4         >/dev/null || missing+=(e2fsprogs)
[ -d /usr/share/keyrings ] && [ -f /usr/share/keyrings/debian-ports-archive-keyring.gpg ] || missing+=(debian-ports-archive-keyring)
if [ ${#missing[@]} -gt 0 ]; then
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing[@]}"
fi

mkdir -p "${BOOT}" "${ROOTFS}"

# ---------- 2/4 debootstrap ----------
log "2/4 debootstrap riscv64 (first stage)"
if [ ! -f "${ROOTFS}/debootstrap/debootstrap.log" ]; then
  sudo debootstrap --arch=riscv64 --variant=minbase \
    --include=systemd,udev,openssh-server,iproute2,initramfs-tools,linux-image-riscv64,sudo,ca-certificates \
    --keyring /usr/share/keyrings/debian-ports-archive-keyring.gpg \
    sid "${ROOTFS}" "${MIRROR}"
fi

log "2b/4 second stage (runs riscv64 binaries via qemu-user-static/binfmt)"
sudo chroot "${ROOTFS}" /debootstrap/debootstrap --second-stage

# ---------- 3/4 guest basic setup ----------
log "3/4 guest setup (password, user, serial console, initramfs)"
sudo chroot "${ROOTFS}" /bin/bash -c '
  set -e
  echo "root:bpla" | chpasswd
  id bpla >/dev/null 2>&1 || useradd -m -s /bin/bash bpla
  echo "bpla:bpla" | chpasswd
  systemctl enable serial-getty@ttyS0
  update-initramfs -u
'
sudo cp "$(ls -1 "${ROOTFS}"/boot/vmlinuz-* | tail -1)"   "${BOOT}/vmlinuz"
sudo cp "$(ls -1 "${ROOTFS}"/boot/initrd.img-* | tail -1)" "${BOOT}/initrd"

# ---------- 4/4 bootable disk ----------
log "4/4 build ext4 disk from rootfs"
fallocate -l "${DISK_SIZE}" "${WORK}/rootdisk.img"
sudo mkfs.ext4 -q -F -d "${ROOTFS}" "${WORK}/rootdisk.img"

sudo chown "$USER" "${BOOT}"/* "${WORK}/rootdisk.img"

log "VM assets ready:"
ls -la "${BOOT}" "${WORK}/rootdisk.img"
cat <<EOF

Boot it with:
  bash run_vm_riscv64.sh "${WORK}"
Guest login: root/bpla  (or bpla/bpla)
EOF