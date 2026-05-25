#!/usr/bin/env nix-shell
#!nix-shell -i bash -p sshfs
# Mount HPC scratch via sshfs.
# Run from any directory — mounts into /tmp.
# Unmount with: fusermount -u /tmp/sshFH-HPC-$USER   (or: umount on macOS)

set -euo pipefail

MOUNT_DIR="${TMPDIR:-/tmp}/sshFH-HPC-${USER:-$(id -un)}"

mkdir -p "$MOUNT_DIR"

# Already mounted?
if mountpoint -q "$MOUNT_DIR" 2>/dev/null; then
    echo "Already mounted at $MOUNT_DIR"
    exit 0
fi

sshfs unibo:/scratch.hpc/joaofilipe.silvade/ "$MOUNT_DIR" \
    -o follow_symlinks \
    -o reconnect \
    -o ServerAliveInterval=15 \
    -o idmap=user

echo "Mounted → $MOUNT_DIR"
echo "Unmount:  fusermount -u $MOUNT_DIR"