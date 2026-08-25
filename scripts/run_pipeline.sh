#!/bin/bash
# ---------------------------------------------------------------------------
# fathom END-TO-END pipeline submitter -- run this on a LOGIN node.
#
# One command does everything, from a clean checkout:
#   0. download MuSiQue + 2Wiki + HotpotQA raw data if missing (needs internet ->
#      login node; the compute jobs cannot do this themselves),
#   1. sbatch run_musique.sh   (GPU)  --+
#   2. sbatch run_2wiki.sh     (GPU)  --+-- submitted together, run in PARALLEL
#   3. sbatch run_hotpot.sh    (GPU)  --+
#   4. sbatch run_transfer.sh  (CPU) x1 -- ONE fused 2-train/1-test job gated
#                                        (afterok) on ALL THREE extractions. It
#                                        loops over every combination that trains
#                                        on two datasets and tests on the held-out
#                                        third (3 combos for these datasets), so
#                                        the whole sweep needs all three runs and
#                                        waits for all three to finish.
#
# If ANY extraction job fails, --kill-on-invalid-dep=yes auto-CANCELS the transfer
# job (a 2-train/1-test sweep cannot run without all three). Fix the failure
# (logs in outputs/slurm_logs/) and resubmit -- extraction RESUMES, skipping
# questions whose activations already exist.
#
# Usage:
#   bash scripts/run_pipeline.sh          # from anywhere; cds itself
#
# Tunables propagate into the jobs via the environment, e.g.:
#   PER_CLASS=500 bash scripts/run_pipeline.sh
#   MODEL=meta-llama/Llama-3.1-8B-Instruct bash scripts/run_pipeline.sh   # whole pipeline on another model
# ---------------------------------------------------------------------------
set -euo pipefail

# The SBATCH scripts MUST be submitted from scripts/ -- their --chdir and -o
# paths are relative to the submission directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# MODEL / MODEL_SLUG -- exported so every sbatch'd job inherits the SAME model id
# and path namespace (outputs/<slug>/ + figures/<slug>/). Override: MODEL=<hf-id>.
source "$SCRIPT_DIR/model_env.sh"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "[pipeline] ERROR: sbatch not found -- run this on the cluster (login node)."
    exit 1
fi

MUSIQUE_RAW="../data/raw/musique/musique_ans_v1.0_dev.jsonl"
WIKI_RAW="../data/raw/2wiki/dev.json"
HOTPOT_RAW="../data/raw/hotpot/hotpot_train_v1.1.json"   # train: HotpotQA 'level' varies only here

# ---- 0) raw datasets (skipped when already present; delete to re-download) ----
if [ -f "$MUSIQUE_RAW" ]; then
    echo "[pipeline] MuSiQue raw data present: $MUSIQUE_RAW"
else
    echo "[pipeline] MuSiQue raw data missing -> downloading ..."
    if ! command -v python >/dev/null 2>&1; then
        echo "[pipeline] ERROR: the MuSiQue download needs python (gdown)."
        echo "           Activate the env first:  source scripts/set_up_env.sh"
        exit 1
    fi
    bash download_musique.sh
fi

if [ -f "$WIKI_RAW" ]; then
    echo "[pipeline] 2Wiki raw data present: $WIKI_RAW"
else
    echo "[pipeline] 2Wiki raw data missing -> downloading ..."
    bash download_2wiki.sh
fi

if [ -f "$HOTPOT_RAW" ]; then
    echo "[pipeline] HotpotQA raw data present: $HOTPOT_RAW"
else
    echo "[pipeline] HotpotQA raw data missing -> downloading the TRAIN split ..."
    # HotpotQA's easy/medium/hard 'level' varies only in train. The HF parquet path
    # needs pyarrow, which on Alliance ships via the Arrow module -- load it
    # best-effort so the download doesn't fall back to the rate-limited /rows API.
    module load gcc arrow 2>/dev/null || true
    HOTPOT_SPLIT=train bash download_hotpot.sh
fi

# SLURM creates log FILES but not parent DIRS -- after a fresh clone/cleanup
# the jobs would die before writing anything if this dir is missing.
mkdir -p ../outputs/slurm_logs

# ---- 1+2+3) the three independent GPU extraction runs, in parallel ----
jid_musique="$(sbatch --parsable run_musique.sh)"
jid_musique="${jid_musique%%;*}"
echo "[pipeline] submitted run_musique.sh  -> job ${jid_musique}"

jid_wiki="$(sbatch --parsable run_2wiki.sh)"
jid_wiki="${jid_wiki%%;*}"
echo "[pipeline] submitted run_2wiki.sh    -> job ${jid_wiki}"

jid_hotpot="$(sbatch --parsable run_hotpot.sh)"
jid_hotpot="${jid_hotpot%%;*}"
echo "[pipeline] submitted run_hotpot.sh   -> job ${jid_hotpot}"

# ---- 4) cross-dataset FUSED transfer: 2-train/1-test over ALL combinations ----
# Each combination trains on TWO datasets' extreme classes and tests on the
# held-out THIRD, so the single sweep needs ALL THREE extractions. Submit ONE
# CPU job gated (afterok) on all three GPU runs; run_transfer.sh then loops over
# every 2-train/1-test combination internally (3 for these datasets) and writes
# figures/transfer_sweep/transfer_<train_a>_<train_b>--<test>/ per combo.
# Resolve the run dirs from the SAME override vars the dataset jobs honor.
musique_out="${OUT:-outputs/${MODEL_SLUG}/run_musique}"
wiki_out="${WIKI_OUT:-outputs/${MODEL_SLUG}/run_2wiki}"
hotpot_out="${HOTPOT_OUT:-outputs/${MODEL_SLUG}/run_hotpot}"

# RUNS is read by run_transfer.sh (propagated via the environment, like the other
# tunables). To add a dataset, append its extraction job id to the afterok list
# and its run dir to RUNS -- every 2-train/1-test combination then appears
# automatically inside the sweep.
jid_transfer="$(RUNS="${wiki_out} ${hotpot_out} ${musique_out}" sbatch --parsable \
    --kill-on-invalid-dep=yes \
    --dependency="afterok:${jid_wiki}:${jid_hotpot}:${jid_musique}" run_transfer.sh)"
jid_transfer="${jid_transfer%%;*}"
echo "[pipeline] submitted run_transfer.sh (fused 2-train/1-test, all combos) -> job ${jid_transfer}"
echo "[pipeline]   gated afterok on: ${jid_wiki} (2wiki) + ${jid_hotpot} (hotpot) + ${jid_musique} (musique)"

echo "[pipeline] all submitted."
echo "[pipeline]   model:    ${MODEL}  (slug: ${MODEL_SLUG})"
echo "[pipeline]   watch:    squeue -u \$USER"
echo "[pipeline]   logs:     outputs/slurm_logs/"
echo "[pipeline]   figures:  figures/${MODEL_SLUG}/run_musique/  figures/${MODEL_SLUG}/run_2wiki/  figures/${MODEL_SLUG}/run_hotpot/"
echo "[pipeline]   transfer (fused 2-train/1-test): figures/${MODEL_SLUG}/transfer_sweep/transfer_musique_hotpot--2wiki/  transfer_2wiki_hotpot--musique/  transfer_2wiki_musique--hotpot/"
