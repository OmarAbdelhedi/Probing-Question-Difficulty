#!/bin/bash
#SBATCH -J fathom-transfer
#SBATCH --mem=32GB
#SBATCH -t 0-02:59:00
#SBATCH --mail-type=NONE
#SBATCH --chdir=../src/
#SBATCH -o ../outputs/slurm_logs/fathom_transfer_%j.out
#SBATCH --account=def-yassine

# ---------------------------------------------------------------------------
# fathom CROSS-DATASET FUSED TRANSFER analysis (2-train / 1-test) -- CPU-only,
# NO GPU.
#
# For EVERY way to hold ONE finished run out as the test set, the low-vs-high
# depth axis is trained on the FUSED pool of the OTHER runs' extreme classes and
# evaluated on the held-out run -- against that run's within-dataset CV ceiling
# (headline metric: retention), plus cross-dataset interpolation of any middle
# classes, for BOTH a raw and a surface-residualized axis. With the default
# three runs (MuSiQue, 2Wiki, HotpotQA) that is the 3 combinations:
#     musique + hotpot  -> 2wiki
#     2wiki   + hotpot  -> musique
#     2wiki   + musique -> hotpot
# Pooling two SOURCE datasets is stronger evidence than any single 1-vs-1 pair
# that the depth axis is dataset-GENERAL rather than template-specific -- and
# fusing HotpotQA's HUMAN-annotated 'level' axis with the STRUCTURAL hop-count
# axis (MuSiQue / 2Wiki) tests whether those two notions of difficulty share a
# direction.
#
# Each combination writes its OWN dir with FIXED-name artifacts (no clash), under
# the per-model FIGS_ROOT (figures/<slug>/ by default, see model_env.sh below):
#   <FIGS_ROOT>/transfer_sweep/transfer_<train_a>_<train_b>--<test>/
#       fused_transfer_report.png / .json
#       fused_transfer_summary.json / fused_transfer_residual_summary.json
# plus one index <FIGS_ROOT>/transfer_sweep/fused_transfer_sweep_summary.json.
#
# This is a THIN wrapper around `python transfer_suite.py fused`, which owns the
# whole sweep (enumerating the 2-train/1-test combinations, per-combo training +
# scoring, and plotting). The script only resolves the READY runs and hands them
# over. Label vocabularies do NOT need to match across runs -- each run is
# ordered by its own labels and the axis maps low<->low / high<->high, so
# MuSiQue's 2hop/3hop/4hop, 2Wiki's easy/hard, and HotpotQA's easy/medium/hard
# all interoperate -- but the suite hard-errors if the runs disagree on model /
# pooling / layers.
#
# PREREQ: each run must already exist (run_musique.sh / run_2wiki.sh /
# run_hotpot.sh). Because every combination trains on two runs and tests on a
# third, the sweep needs >=3 READY runs; with fewer ready it SOFT-skips (clear
# message, exit 0) so a partial pipeline never fails this job.
#
# Submit from scripts/:   cd scripts && sbatch run_transfer.sh
#
# Tunables -- override at submit time, e.g.:
#   RUNS="outputs/run_musique outputs/run_2wiki outputs/run_hotpot" sbatch run_transfer.sh
#   RUNS="outputs/a outputs/b outputs/c outputs/d" sbatch run_transfer.sh  # any N>=3 -> all 2-train/1-test combos
#   FIGS_ROOT=figures/<slug> sbatch run_transfer.sh                        # parent of transfer_sweep/
#   MODEL=<hf-id> sbatch run_transfer.sh                                   # fuse a DIFFERENT model's runs
# MODEL_SLUG namespaces the defaults so each model fuses ITS OWN runs into ITS
# OWN figures/<slug>/transfer_sweep/ (the suite still hard-errors if the resolved
# runs disagree on model/pooling/layers).
source ../scripts/model_env.sh
RUNS="${RUNS:-outputs/${MODEL_SLUG}/run_2wiki outputs/${MODEL_SLUG}/run_hotpot outputs/${MODEL_SLUG}/run_musique}"
FIGS_ROOT="${FIGS_ROOT:-figures/${MODEL_SLUG}}"
# ---------------------------------------------------------------------------

source ../scripts/helper.sh
source ../scripts/set_up_env.sh
cd ..

# Split RUNS (whitespace-separated) into an array.
read -r -a RUN_ARR <<< "$RUNS"

# A run is "ready" iff it has the artifacts the suite loads: activations/ + metadata.jsonl.
run_ready() { [ -d "$1/activations" ] && [ -f "$1/metadata.jsonl" ]; }

# Drop runs that are not yet built (with a message) instead of failing the whole
# job, so a partial set still produces every available 2-train/1-test combo.
READY=()
for run in "${RUN_ARR[@]}"; do
    if run_ready "$run"; then
        READY+=("$run")
    else
        echo "[run_transfer] NOT READY (dropped from fusion): $run (need $run/activations + metadata.jsonl)"
    fi
done

n=${#READY[@]}
if [ "$n" -lt 3 ]; then
    echo "[run_transfer] SKIP: 2-train/1-test fusion needs >=3 ready runs; only ${n} ready: ${READY[*]:-<none>}"
    exit 0
fi

echo "[run_transfer] fused 2-train/1-test over ${n} ready run(s): ${READY[*]}"
echo "[run_transfer] figures -> ${FIGS_ROOT}/transfer_sweep/"
python transfer_suite.py fused --runs "${READY[@]}" --plots-root "$FIGS_ROOT"
status=$?
echo "[run_transfer] transfer_suite.py fused exited with status ${status}"
exit ${status}
