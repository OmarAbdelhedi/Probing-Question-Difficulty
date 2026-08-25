#!/bin/bash
#SBATCH -J fathom-replot
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH -t 0-02:59:00
#SBATCH --mail-type=NONE
#SBATCH --chdir=../src/
#SBATCH -o ../outputs/slurm_logs/fathom_replot_%A_%a.out
#SBATCH --account=def-yassine
#SBATCH --array=0-8

# ---------------------------------------------------------------------------
# fathom RE-PLOT -- CPU-only, NO GPU, NO re-extraction. PARALLEL job array.
#
# Re-render EVERY figure (Tier 2 / 2b / 2g per run + the fused cross-dataset
# transfer report) from the EXISTING run artifacts (outputs/<slug>/run_*/
# activations + metadata.jsonl), using the current plot_* code. Use this after
# editing the plotting functions (titles / axis labels / styling) to refresh
# figures/<slug>/... WITHOUT re-running the expensive GPU extraction jobs.
#
# WHY AN ARRAY (was a serial double-for-loop): the wall-clock here is the CPU
# probe recomputation, dominated by Tier 2b's PERMS-shuffle permutation null
# (it re-fits the probe across every layer/fold PERMS times per slug x dataset).
# The 3 slugs x 3 datasets = 9 combos are INDEPENDENT, so we fan them out across
# 9 array tasks: wall-clock collapses from "sum of all 9" to "slowest single
# combo". analyze.py + transfer_suite.py are pure numpy + matplotlib (no GPU);
# OMP_NUM_THREADS=1 (set in set_up_env.sh) keeps each task single-threaded, so
# --cpus-per-task=1 is the honest request and schedules best.
#
# TWO PHASES, ONE FILE (controlled by $REPLOT_PHASE + the array index):
#   tiers (default, --array=0-8): each task = ONE (slug, dataset), runs Tier
#       2/2b/2g for it. Task 0 also submits the fused phase below, held until
#       ALL tier tasks finish (the fused report needs all 3 of a slug's runs).
#   fused (--array=0-2, submitted by task 0): each task = ONE slug's fused
#       cross-dataset transfer over its 3 runs.
# The dependency is afterANY (not afterok): the fused report reads the run
# ACTIVATIONS, not the tier outputs, so a failed/flaky tier task must not block
# it -- transfer_suite.py soft-skips any run that is not ready. Switch to
# afterok below if you want strict "all tiers green first" semantics.
#
# Submit from scripts/:   cd scripts && sbatch replot.sh
#   (one submission; task 0 wires up the dependent fused array automatically)
#
# Override the sweep at submit time, e.g.:
#   SLUGS="gemma27b" DATASETS="run_musique" sbatch --array=0 replot.sh  # 1 combo
#   TIERS="2" sbatch replot.sh                          # only the Tier-2 headline
#   PERMS=100 sbatch replot.sh                          # cheaper Tier-2b null
# NOTE: if you narrow SLUGS/DATASETS you MUST also pass a matching --array
# (size = #slugs * #datasets, indices row-major slug-then-dataset); the fused
# phase always re-derives --array from #slugs.
# ---------------------------------------------------------------------------
set -uo pipefail

source ../scripts/helper.sh
source ../scripts/set_up_env.sh
cd ..

# Read the sweep from the ENVIRONMENT as plain strings (so user overrides
# propagate verbatim to the fused resubmit via --export=ALL), then expand into
# LOCAL arrays named _SLUGS/_DATASETS -- deliberately NOT reassigning SLUGS /
# DATASETS, which would drop the exported env strings the fused phase relies on.
_SLUGS=(${SLUGS:-gemma27b mistral24b qwen32b})
_DATASETS=(${DATASETS:-run_2wiki run_hotpot run_musique})
# 2g skips gracefully where there are <3 ordered classes (e.g. 2wiki hops_binary).
TIERS="${TIERS:-2 2b 2g}"
PERMS="${PERMS:-200}"
PHASE="${REPLOT_PHASE:-tiers}"

n_slugs=${#_SLUGS[@]}
n_ds=${#_DATASETS[@]}
tid="${SLURM_ARRAY_TASK_ID:-0}"

if [ "${PHASE}" = "fused" ]; then
    # -------- FUSED phase: one task per slug -------------------------------
    if [ "${tid}" -ge "${n_slugs}" ]; then
        echo "[replot] fused task ${tid} out of range (only ${n_slugs} slug(s)); nothing to do."
        exit 0
    fi
    slug="${_SLUGS[${tid}]}"
    runs=""
    for ds in "${_DATASETS[@]}"; do
        runs="${runs} outputs/${slug}/${ds}"
    done
    echo "[replot] FUSED task ${tid}: ${slug} ->${runs} -> figures/${slug}/transfer_sweep/"
    python transfer_suite.py fused --runs ${runs} --plots-root "figures/${slug}"
    echo "[replot]   fused exited with status $?"
    echo "[replot] FUSED task ${tid} DONE (${slug})."
    exit 0
fi

# -------- TIERS phase: one task per (slug, dataset) ------------------------
# Task 0 submits the dependent fused array (one task per slug), held until the
# WHOLE tier array finishes. Only task 0 does this, so it runs exactly once;
# guarded on SLURM_ARRAY_JOB_ID so a local `bash replot.sh` test never sbatches.
if [ "${tid}" -eq 0 ] && [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    last_slug=$(( n_slugs - 1 ))
    echo "[replot] task 0: submitting dependent fused array (--array=0-${last_slug}, afterany:${SLURM_ARRAY_JOB_ID}) ..."
    sbatch --array=0-${last_slug} \
           --dependency=afterany:${SLURM_ARRAY_JOB_ID} \
           --export=ALL,REPLOT_PHASE=fused \
           scripts/replot.sh \
        || echo "[replot]   WARNING: could not submit fused array (run scripts/run_transfer.sh per model by hand)."
fi

slug_idx=$(( tid / n_ds ))
ds_idx=$(( tid % n_ds ))
if [ "${slug_idx}" -ge "${n_slugs}" ]; then
    echo "[replot] task ${tid} -> slug index ${slug_idx} out of range (only ${n_slugs} slug(s)); narrow --array to #slugs*#datasets. Nothing to do."
    exit 0
fi
slug="${_SLUGS[${slug_idx}]}"
ds="${_DATASETS[${ds_idx}]}"
out="outputs/${slug}/${ds}"
figs="figures/${slug}/${ds}"

echo "[replot] TASK ${tid}: ${slug}/${ds} tiers '${TIERS}' -> ${figs}/"
if [ ! -d "${out}/activations" ]; then
    echo "[replot] SKIP ${out} (no activations/ -- run the GPU job first)"
    exit 0
fi
for tier in ${TIERS}; do
    echo "[replot] ${slug}/${ds} tier ${tier} -> ${figs}/ ..."
    if [ "${tier}" = "2b" ]; then
        python analyze.py --output "${out}" --tier 2b --plots-dir "${figs}" --perms "${PERMS}"
    else
        python analyze.py --output "${out}" --tier "${tier}" --plots-dir "${figs}"
    fi
    echo "[replot]   tier ${tier} exited with status $?"
done
echo "[replot] TASK ${tid} DONE (${slug}/${ds})."
