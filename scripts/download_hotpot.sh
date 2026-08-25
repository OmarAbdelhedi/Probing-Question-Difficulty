#!/bin/bash
# ---------------------------------------------------------------------------
# Download HotpotQA into data/raw/hotpot/ (where run_hotpot.sh / the converter
# expect it: data/raw/hotpot/hotpot_dev_distractor_v1.json).
#
# HotpotQA (Yang et al. 2018) is obtained from one of two sources, tried in order
# (override with HOTPOT_SOURCE):
#   1. curtis.ml.cmu.edu -- the OFFICIAL host (https://hotpotqa.github.io). Plain
#      .json, no python deps. BUT it is HTTP-only (port 80) and frequently down or
#      unreachable from HPC networks that only allow outbound 443 -- so it may
#      time out. We give it a SHORT timeout, then fall back to:
#   2. HuggingFace Hub (hotpotqa/hotpot_qa), all HTTPS (port 443). Two ways:
#      a. PREFERRED -- download the split's parquet file(s) straight from the Hub
#         CDN (ONE request per shard, not rate-limited) and convert with pyarrow.
#         On Alliance Canada pyarrow is NOT pip-installable; load it with
#         `module load gcc arrow` before running.
#      b. FALLBACK (no pyarrow) -- the dataset-viewer /rows JSON API, pure stdlib.
#         Works off-cluster, but it caps at 100 rows/request and RATE-LIMITS shared
#         cluster IPs (HTTP 429), so it is unreliable on a busy login node -- prefer
#         (a) there.
#      Both keep the fields the builder labels on (question / id / level / type /
#      supporting_facts) and OMIT the bulky 'context' passages, which FATHOM never
#      uses (skipping them also keeps the 90k-row train conversion memory-safe).
#
# IMPORTANT:
#   * Run this on a LOGIN NODE -- SLURM/compute nodes on most clusters (incl.
#     Alliance Canada) have NO outbound internet, so the download will hang/fail
#     there. Download once on a login node, then submit run_hotpot.sh.
#
# Usage:
#   bash scripts/download_hotpot.sh
#   module load gcc arrow && HOTPOT_SOURCE=hf bash scripts/download_hotpot.sh  # cluster: parquet path
#   HOTPOT_SOURCE=hf     bash scripts/download_hotpot.sh   # skip curtis (e.g. port 80 blocked)
#   HOTPOT_SOURCE=curtis bash scripts/download_hotpot.sh   # official host only, no HF fallback
#   HOTPOT_SPLIT=train   bash scripts/download_hotpot.sh   # dev_distractor (default)|dev_fullwiki|train|test
#   HOTPOT_URL=<url>     bash scripts/download_hotpot.sh   # force a direct .json url (curtis-style source)
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/data/raw/hotpot"
mkdir -p "$DEST"

HOTPOT_SPLIT="${HOTPOT_SPLIT:-train}"         # train: HotpotQA's 'level' varies ONLY here (dev/test are all 'hard')
HOTPOT_SOURCE="${HOTPOT_SOURCE:-auto}"        # auto | curtis | hf

# Map the split -> curtis filename AND the HF (config, split). The TRAIN split is
# the default because the FATHOM probe labels on 'level' (easy/medium/hard), which
# is constant ('hard') on dev/distractor and test -- only train carries the contrast.
case "$HOTPOT_SPLIT" in
    dev_distractor)     FILE="hotpot_dev_distractor_v1.json"; HF_CONFIG="distractor"; HF_SPLIT="validation" ;;
    dev_fullwiki)       FILE="hotpot_dev_fullwiki_v1.json";   HF_CONFIG="fullwiki";   HF_SPLIT="validation" ;;
    train)              FILE="hotpot_train_v1.1.json";        HF_CONFIG="distractor"; HF_SPLIT="train" ;;
    test|test_fullwiki) FILE="hotpot_test_fullwiki_v1.json";  HF_CONFIG="fullwiki";   HF_SPLIT="test" ;;
    *)
        echo "[hotpot] ERROR: unknown HOTPOT_SPLIT='$HOTPOT_SPLIT'"
        echo "         choose one of: dev_distractor | dev_fullwiki | train | test"
        exit 1 ;;
esac
JSON="$DEST/$FILE"

# Valid-payload check: a HotpotQA json file is a JSON array, so the first
# non-whitespace byte must be '['. Catches dead links that serve an HTML page.
is_json_array() { [ -s "$1" ] && [ "$(head -c 64 "$1" | tr -d '[:space:]' | head -c 1)" = "[" ]; }

# ---- source 1: curtis.ml.cmu.edu (official, HTTP / port 80) ----
try_curtis() {
    local base="http://curtis.ml.cmu.edu/datasets/hotpot" url="${HOTPOT_URL:-}"
    if [ -z "$url" ]; then                          # auto-detect from the homepage (HTTPS)
        local file_re="${FILE//./\\.}"
        local page
        for page in "https://hotpotqa.github.io/" "https://hotpotqa.github.io/index.html"; do
            url="$(curl -fsSL "$page" 2>/dev/null | grep -oE "https?://[^\"' )]*${file_re}" | head -n1 || true)"
            [ -n "$url" ] && break
        done
    fi
    url="${url:-$base/$FILE}"
    echo "[hotpot] (curtis) GET $url"
    # SHORT connect-timeout: a blocked/dead port 80 should fall back to HF fast.
    if curl -fL --retry 2 --connect-timeout 10 -o "$JSON" "$url" && is_json_array "$JSON"; then
        return 0
    fi
    rm -f "$JSON"
    return 1
}

# ---- source 2a: HuggingFace parquet via the Hub CDN (one request per shard) ----
# Direct file downloads (NOT the rate-limited dataset-viewer API), converted with
# pyarrow. Reads only the columns FATHOM needs (skips the bulky 'context') so the
# 90k-row train split is memory-safe, and rebuilds list-of-pairs supporting_facts.
hf_via_parquet() {
    local base="https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/${HF_CONFIG}/${HF_SPLIT}"
    local tmp="$DEST/_parquet"; rm -rf "$tmp"; mkdir -p "$tmp"
    local i=0
    while : ; do                                    # shards 0,1,2,... until one 404s
        echo "[hotpot] (hf) GET parquet shard $i ..."
        if curl -fL --retry 3 -o "$tmp/${i}.parquet" "${base}/${i}.parquet"; then
            i=$((i + 1))
        else
            rm -f "$tmp/${i}.parquet"
            break
        fi
    done
    if [ "$i" -eq 0 ]; then
        echo "[hotpot] (hf) no parquet shard downloaded."
        rm -rf "$tmp"
        return 1
    fi

    HOTPOT_PARQUET_DIR="$tmp" HOTPOT_JSON_OUT="$JSON" python - <<'PY'
import glob, json, os
import pyarrow.parquet as pq

tmp = os.environ["HOTPOT_PARQUET_DIR"]
out_path = os.environ["HOTPOT_JSON_OUT"]
# Read ONLY the columns FATHOM needs -- skip the bulky 'context' so the 90k-row
# train split converts without exhausting login-node memory (parquet is columnar,
# so projected columns are the only ones read from disk).
NEEDED = ["id", "question", "answer", "type", "level", "supporting_facts"]
rows = []
for p in sorted(glob.glob(os.path.join(tmp, "*.parquet"))):
    avail = set(pq.read_schema(p).names)
    rows.extend(pq.read_table(p, columns=[c for c in NEEDED if c in avail]).to_pylist())

out = []
for ex in rows:
    sf = ex.get("supporting_facts") or {"title": [], "sent_id": []}
    out.append({
        "_id": ex.get("id", ""),
        "question": ex.get("question", ""),
        "answer": ex.get("answer", ""),
        "type": ex.get("type", ""),
        "level": ex.get("level", ""),
        "supporting_facts": [[t, int(s)] for t, s in zip(sf.get("title", []), sf.get("sent_id", []))],
    })
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(out, fh)
print(f"[hotpot] (hf) converted {len(out)} examples (context omitted) -> {out_path}")
PY
    local st=$?
    rm -rf "$tmp"
    if [ "$st" -eq 0 ] && is_json_array "$JSON"; then
        return 0
    fi
    rm -f "$JSON"
    return 1
}

# ---- source 2b: HuggingFace dataset-viewer /rows JSON API (pure stdlib) ----
# No parquet/pyarrow. Off-cluster only in practice: the API rate-limits shared
# cluster IPs (HTTP 429). Omits the bulky 'context' (FATHOM never uses it).
hf_via_rows() {
    echo "[hotpot] (hf) fetching ${HF_CONFIG}/${HF_SPLIT} via the dataset-viewer /rows API ..."
    HOTPOT_HF_CONFIG="$HF_CONFIG" HOTPOT_HF_SPLIT="$HF_SPLIT" HOTPOT_JSON_OUT="$JSON" python - <<'PY'
import json, os, time, urllib.error, urllib.request

config, split = os.environ["HOTPOT_HF_CONFIG"], os.environ["HOTPOT_HF_SPLIT"]
out_path = os.environ["HOTPOT_JSON_OUT"]
DATASET = "hotpotqa/hotpot_qa"
BASE = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # the /rows API caps length at 100

def fetch(offset, length):
    url = (f"{BASE}?dataset={DATASET}&config={config}&split={split}"
           f"&offset={offset}&length={length}")
    last = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:       # 429 -> respect Retry-After, long backoff
            last = exc
            wait = int(exc.headers.get("Retry-After") or 0) if exc.code == 429 else 0
            time.sleep(max(wait, 5 * (attempt + 1)))
        except Exception as exc:                    # transient network -> backoff & retry
            last = exc
            time.sleep(5 * (attempt + 1))
    raise last

total = fetch(0, 1)["num_rows_total"]
print(f"[hotpot] (hf) {total} rows to fetch ...")
out, off = [], 0
while off < total:
    rows = fetch(off, PAGE).get("rows", [])
    if not rows:
        break
    for item in rows:
        ex = item.get("row", {})
        sf = ex.get("supporting_facts") or {"title": [], "sent_id": []}
        out.append({
            "_id": ex.get("id", ""),
            "question": ex.get("question", ""),
            "answer": ex.get("answer", ""),
            "type": ex.get("type", ""),
            "level": ex.get("level", ""),
            "supporting_facts": [[t, int(s)] for t, s in
                                 zip(sf.get("title", []), sf.get("sent_id", []))],
        })
    off += len(rows)
    time.sleep(0.5)                                 # be gentle on the rate limiter
    print(f"[hotpot] (hf)   {off}/{total}", end="\r", flush=True)
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(out, fh)
print(f"\n[hotpot] (hf) wrote {len(out)} examples -> {out_path}")
PY
    local st=$?
    if [ "$st" -eq 0 ] && is_json_array "$JSON"; then
        return 0
    fi
    rm -f "$JSON"
    return 1
}

# ---- source 2: HuggingFace dispatcher (parquet if pyarrow, else /rows) ----
try_hf() {
    if ! command -v python >/dev/null 2>&1; then
        echo "[hotpot] (hf) python not found -- cannot fetch the HF mirror."
        echo "         Activate your env first (e.g. source scripts/set_up_env.sh)."
        return 1
    fi
    if python -c "import pyarrow" >/dev/null 2>&1; then
        if hf_via_parquet; then return 0; fi
        echo "[hotpot] (hf) parquet path failed; trying the /rows API ..."
    else
        echo "[hotpot] (hf) pyarrow not importable -- the fast parquet path is unavailable."
        echo "             On Alliance Canada load it (NOT via pip):  module load gcc arrow"
        echo "             Trying the /rows API meanwhile (may hit HTTP 429 on a shared cluster IP) ..."
    fi
    hf_via_rows
}

echo "[hotpot] split '$HOTPOT_SPLIT' -> $FILE  (source: $HOTPOT_SOURCE)"
ok=0
if [ "$HOTPOT_SOURCE" = "curtis" ] || [ "$HOTPOT_SOURCE" = "auto" ]; then
    if try_curtis; then ok=1; fi
fi
if [ "$ok" -ne 1 ] && { [ "$HOTPOT_SOURCE" = "hf" ] || [ "$HOTPOT_SOURCE" = "auto" ]; }; then
    [ "$HOTPOT_SOURCE" = "auto" ] && echo "[hotpot] curtis unavailable -> falling back to the HuggingFace mirror ..."
    if try_hf; then ok=1; fi
fi

if [ "$ok" -ne 1 ]; then
    echo "[hotpot] ERROR: could not obtain $FILE from any source."
    echo "  * curtis.ml.cmu.edu is HTTP-only and often down / blocked (many HPC nets allow only 443)."
    echo "  * For the reliable HF parquet path, load pyarrow first, then force HF:"
    echo "      module load gcc arrow && HOTPOT_SOURCE=hf bash scripts/download_hotpot.sh"
    echo "    (the /rows API fallback can return HTTP 429 on a shared cluster IP)."
    echo "  * Or fetch the file manually and place the json at: $JSON"
    exit 1
fi

bytes="$(wc -c < "$JSON" | tr -d ' ')"
echo "[hotpot] done -- $FILE (${bytes} bytes) in $DEST"
if [ "$HOTPOT_SPLIT" = "train" ]; then
    echo "[hotpot] ready. run_hotpot.sh default expects: $JSON"
else
    echo "[hotpot] NOTE: this is the '$HOTPOT_SPLIT' split; point HOTPOT_JSON at it:"
    echo "        HOTPOT_JSON=$JSON sbatch run_hotpot.sh"
    case "$HOTPOT_SPLIT" in
        dev_distractor|dev_fullwiki)
            echo "[hotpot] WARNING: dev is entirely level='hard' -- the default LABEL_SCHEME=level needs >=2"
            echo "         levels, so use the train split (the default) or LABEL_SCHEME=type on this split." ;;
        test|test_fullwiki)
            echo "[hotpot] WARNING: the test split has no 'level'/'type' -- it will NOT build a labeled jsonl." ;;
    esac
fi
