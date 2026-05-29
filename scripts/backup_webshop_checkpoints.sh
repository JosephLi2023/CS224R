#!/usr/bin/env bash
# Download SFT warm-start (latest) + all attention round adapters from Modal.
#
# READ-ONLY on Modal: only `modal volume get` (nothing deleted on the volume).
#
# Usage:
#   bash scripts/backup_webshop_checkpoints.sh
#   DEST=~/my_backup bash scripts/backup_webshop_checkpoints.sh
#
# Layout:
#   $DEST/checkpoints/sft_webshop_v3_mlpr32_.../latest/
#   $DEST/checkpoints/webshop_attention_v1_seed11_round00_adapter/
#   ... round09_adapter/

set -euo pipefail

VOLUME="${VOLUME:-cs224r-hgpo-vol}"
DEST="${DEST:-$HOME/CS224R_checkpoints_backup}"

SFT_RUN="checkpoints/sft_webshop_v3_mlpr32_20260527_083426_20260527_153446"
SFT_LATEST="${SFT_RUN}/latest"
ATTENTION_PREFIX="checkpoints/webshop_attention_v1_seed11_round"

clear_empty_local_dir() {
  local path="$1"
  if [[ -d "$path" ]] && [[ -z "$(ls -A "$path" 2>/dev/null)" ]]; then
    echo "  (removing empty local folder from a prior failed download: $path)"
    rmdir "$path"
  fi
}

mkdir -p "$DEST"

echo "=== Modal volume: read-only download (nothing deleted on cloud) ==="
echo ""

clear_empty_local_dir "$DEST/checkpoints/${SFT_LATEST}"

echo "=== SFT warm-start (final adapter: latest/) ==="
echo "  remote: /$SFT_LATEST"
echo "  local:  $DEST/"
modal volume get "$VOLUME" "/$SFT_LATEST" "$DEST" --force

echo ""
echo "=== Attention adapters (rounds 00–09) ==="
for r in 00 01 02 03 04 05 06 07 08 09; do
  name="${ATTENTION_PREFIX}${r}_adapter"
  clear_empty_local_dir "$DEST/checkpoints/${name}"
  echo "  -> /$name"
  modal volume get "$VOLUME" "/${name}" "$DEST" --force
done

echo ""
echo "=== Done ==="
echo "Local root: $DEST"
echo ""
echo "Warm-start for RL:"
echo "  $DEST/checkpoints/${SFT_LATEST}/"
echo "Final attention policy:"
echo "  $DEST/checkpoints/webshop_attention_v1_seed11_round09_adapter/"
echo ""
du -sh "$DEST/checkpoints"/* 2>/dev/null || true
