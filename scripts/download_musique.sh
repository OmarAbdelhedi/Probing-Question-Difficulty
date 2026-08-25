#!/bin/bash
# ---------------------------------------------------------------------------
# Download MuSiQue into data/raw/musique/ (where run_musique.sh / the converter
# expect it: data/raw/musique/musique_ans_v1.0_dev.jsonl, ..._train.jsonl, etc.).
#
# MuSiQue is distributed as a Google-Drive zip via the official repo
# (https://github.com/StonyBrookNLP/musique). This script fetches it with gdown.
#
# IMPORTANT:
#   * Run this on a LOGIN NODE -- SLURM/compute nodes on most clusters (incl.
#     Alliance Canada) have NO outbound internet, so the download will hang/fail
#     there. Do the download once on a login node, then submit run_musique.sh.
#   * Run it with your project env active (so `python`/`pip` resolve), e.g.
#     after `source scripts/set_up_env.sh`.
#
# Usage:
#   bash scripts/download_musique.sh
#   MUSIQUE_GDRIVE_ID=<id> bash scripts/download_musique.sh   # override the Drive id
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/data/raw/musique"
mkdir -p "$DEST"

# --- resolve the Google-Drive file id: env override > upstream auto-detect > default ---
DEFAULT_ID="1tGdADlNjnJfTSZcKktNbjV3qQbg5LWxK"
GDRIVE_ID="${MUSIQUE_GDRIVE_ID:-}"
if [ -z "$GDRIVE_ID" ]; then
    for ref in main master; do
        url="https://raw.githubusercontent.com/StonyBrookNLP/musique/${ref}/download_data.sh"
        GDRIVE_ID="$(curl -fsSL "$url" 2>/dev/null | grep -oE '[A-Za-z0-9_-]{30,}' | head -n1 || true)"
        if [ -n "$GDRIVE_ID" ]; then
            echo "[musique] using Drive id auto-detected from upstream ($ref): $GDRIVE_ID"
            break
        fi
    done
fi
GDRIVE_ID="${GDRIVE_ID:-$DEFAULT_ID}"
echo "[musique] Google-Drive file id: $GDRIVE_ID"

# --- ensure gdown is available ---
if ! python -c "import gdown" >/dev/null 2>&1; then
    echo "[musique] installing gdown into the active environment ..."
    python -m pip install --quiet --upgrade gdown
fi

# --- download ---
ZIP="$DEST/musique_v1.0.zip"
echo "[musique] downloading musique_v1.0.zip ..."
if ! python -m gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" -O "$ZIP"; then
    echo "[musique] ERROR: gdown failed for id ${GDRIVE_ID}."
    echo "  * Are you on a login node? (compute nodes have no internet)"
    echo "  * Override the id:  MUSIQUE_GDRIVE_ID=<id> bash scripts/download_musique.sh"
    echo "  * Or grab musique_v1.0.zip manually from https://github.com/StonyBrookNLP/musique"
    echo "    and unzip its *.jsonl files into: $DEST"
    exit 1
fi

# --- unzip + flatten the *.jsonl into DEST (the archive nests them under data/) ---
echo "[musique] unzipping ..."
TMP="$DEST/_unzip"
rm -rf "$TMP"; mkdir -p "$TMP"
unzip -oq "$ZIP" -d "$TMP"

found=0
while IFS= read -r f; do
    mv -f "$f" "$DEST/"
    found=$((found + 1))
done < <(find "$TMP" -name 'musique_*_v1.0_*.jsonl')
rm -rf "$TMP" "$ZIP"

if [ "$found" -eq 0 ]; then
    echo "[musique] ERROR: no musique_*_v1.0_*.jsonl found inside the archive."
    echo "          Unzip musique_v1.0.zip manually and place its *.jsonl into: $DEST"
    exit 1
fi

echo "[musique] done -- $found jsonl file(s) in $DEST:"
ls -1 "$DEST"
DEV="$DEST/musique_ans_v1.0_dev.jsonl"
if [ -f "$DEV" ]; then
    echo "[musique] ready. run_musique.sh default expects: $DEV"
else
    echo "[musique] NOTE: musique_ans_v1.0_dev.jsonl not present; point MUSIQUE_JSON at one of the files above."
fi
