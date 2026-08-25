#!/bin/bash
# ---------------------------------------------------------------------------
# Download 2WikiMultihopQA into data/raw/2wiki/ (where run_2wiki.sh / the
# converter expect it: data/raw/2wiki/dev.json, plus train.json / test.json).
#
# 2Wiki (Ho et al. 2020) is distributed as a Dropbox zip via the official repo
# (https://github.com/Alab-NII/2wikimultihop): data_ids_april7.zip containing
# train/dev/test.json (+ id_aliases.json). Fetched here with plain curl --
# no python/gdown needed.
#
# IMPORTANT:
#   * Run this on a LOGIN NODE -- SLURM/compute nodes on most clusters (incl.
#     Alliance Canada) have NO outbound internet, so the download will hang/fail
#     there. Download once on a login node, then submit run_2wiki.sh.
#
# Usage:
#   bash scripts/download_2wiki.sh
#   WIKI_URL=<url> bash scripts/download_2wiki.sh    # override the zip url
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/data/raw/2wiki"
mkdir -p "$DEST"

# --- resolve the zip url: env override > upstream README auto-detect > default ---
# The QA data (train/dev/test.json) ships as data_ids_april7.zip. CAREFUL: the
# README ALSO links para_with_hyperlink.zip -- a 1.8 GB Wikipedia-paragraph dump
# with NO train/dev/test.json -- and lists it FIRST, so we must pick the data
# archive BY NAME. Grabbing the first .zip downloads 1.8 GB of the wrong thing.
DEFAULT_URL="https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1"
URL="${WIKI_URL:-}"
if [ -z "$URL" ]; then
    for ref in master main; do
        readme="https://raw.githubusercontent.com/Alab-NII/2wikimultihop/${ref}/README.md"
        zips="$(curl -fsSL "$readme" 2>/dev/null \
                | grep -oE 'https://www\.dropbox\.com/[^") ]+\.zip(\?dl=[01])?' \
                | grep -Ev 'para_with_hyperlink' || true)"
        # Prefer the canonical data_ids_april7.zip; else any other data*.zip.
        URL="$(printf '%s\n' "$zips" | grep -E 'data_ids_april7\.zip' | head -n1 || true)"
        if [ -z "$URL" ]; then
            URL="$(printf '%s\n' "$zips" | grep -E '/[^/]*data[^/]*\.zip' | head -n1 || true)"
        fi
        if [ -n "$URL" ]; then
            echo "[2wiki] using zip url auto-detected from upstream ($ref): $URL"
            break
        fi
    done
fi
URL="${URL:-$DEFAULT_URL}"
case "$URL" in                       # dropbox: force a DIRECT download (dl=1)
    *dl=0*) URL="${URL/dl=0/dl=1}" ;;
    *dl=1*) ;;
    *dropbox.com*) URL="${URL}?dl=1" ;;
esac
echo "[2wiki] zip url: $URL"

# --- download ---
ZIP="$DEST/2wiki_data_ids.zip"
echo "[2wiki] downloading (the zip is ~0.7 GB; this can take a few minutes) ..."
if ! curl -fL --retry 3 -o "$ZIP" "$URL"; then
    echo "[2wiki] ERROR: download failed."
    echo "  * Are you on a login node? (compute nodes have no internet)"
    echo "  * Override the url:  WIKI_URL=<url> bash scripts/download_2wiki.sh"
    echo "  * Or grab the data zip manually from https://github.com/Alab-NII/2wikimultihop"
    echo "    and unzip train/dev/test.json into: $DEST"
    exit 1
fi

# A dead/rotated link serves an HTML page, not a zip -- verify before unzipping.
if ! unzip -tq "$ZIP" >/dev/null 2>&1; then
    echo "[2wiki] ERROR: the downloaded file is not a valid zip (dead/rotated link?)."
    echo "        Get the current link from https://github.com/Alab-NII/2wikimultihop and rerun:"
    echo "        WIKI_URL=<url> bash scripts/download_2wiki.sh"
    exit 1
fi

# --- unzip + flatten the *.json into DEST (archive layout varies: flat or nested) ---
echo "[2wiki] unzipping ..."
TMP="$DEST/_unzip"
rm -rf "$TMP"; mkdir -p "$TMP"
unzip -oq "$ZIP" -d "$TMP"

found=0
while IFS= read -r f; do
    mv -f "$f" "$DEST/"
    found=$((found + 1))
done < <(find "$TMP" -name '*.json')

# On failure, KEEP the zip + extracted tree for inspection (don't force a
# multi-GB re-download just to debug). Only clean up once we have the json.
if [ "$found" -eq 0 ]; then
    echo "[2wiki] ERROR: no *.json found inside the archive (kept for inspection):"
    echo "          zip       : $ZIP"
    echo "          extracted : $TMP"
    echo "        This usually means the url pointed at para_with_hyperlink.zip (the 1.8 GB"
    echo "        paragraph dump) instead of the data zip. Grab the data archive ("
    echo "        data_ids_april7.zip) from https://github.com/Alab-NII/2wikimultihop and rerun:"
    echo "        WIKI_URL=https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1 bash scripts/download_2wiki.sh"
    exit 1
fi
rm -rf "$TMP" "$ZIP"

echo "[2wiki] done -- $found json file(s) in $DEST:"
ls -1 "$DEST"
DEV="$DEST/dev.json"
if [ -f "$DEV" ]; then
    echo "[2wiki] ready. run_2wiki.sh default expects: $DEV"
else
    echo "[2wiki] NOTE: dev.json not present; point WIKI_JSON at one of the files above."
fi
