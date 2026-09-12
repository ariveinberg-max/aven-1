#!/usr/bin/env bash
# Off-machine checkpoint backup -- free, using iCloud Drive (already available
# on every Mac, no new signup or cost). Closes a real, standing risk this
# project has already been burned by once: a checkpoint living on only one
# machine's disk (see research/TASKS.md's "Rented-GPU persistence hazard").
#
# If iCloud storage is full, this simply fails to sync (macOS surfaces its
# own "storage almost full" prompt to the user) -- it does not charge
# anything or upgrade any plan on its own.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HOME/Library/Mobile Documents/com~apple~CloudDocs/aven-1-backups"
mkdir -p "$DEST"

if [ -f "$ROOT/checkpoints/latest.pt" ]; then
    cp "$ROOT/checkpoints/latest.pt" "$DEST/latest.pt.tmp" && mv "$DEST/latest.pt.tmp" "$DEST/latest.pt"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Backed up checkpoints/latest.pt"
fi
if [ -f "$ROOT/checkpoints/tokenizer.json" ]; then
    # Atomic tmp+mv, same as latest.pt above -- a direct cp onto the live,
    # iCloud-synced file races with the file-provider daemon's own lock on
    # it ("Resource deadlock avoided", seen recurring in production).
    # Renaming a new temp file into place never needs that lock.
    cp "$ROOT/checkpoints/tokenizer.json" "$DEST/tokenizer.json.tmp" && mv "$DEST/tokenizer.json.tmp" "$DEST/tokenizer.json"
fi
