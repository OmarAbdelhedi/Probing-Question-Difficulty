# ---------------------------------------------------------------------------
# fathom -- the shared body of the per-dataset end-to-end SLURM jobs
# (run_musique.sh / run_2wiki.sh / run_hotpot.sh). SOURCED by them, never
# submitted itself: SLURM reads #SBATCH headers only from the submitted file,
# so each dataset keeps its own header + knobs and delegates everything below
# to this file.
#
# Every job does the same three things:
#   1. build the labeled jsonl (python -m src.datasets.build_${DATASET}) -- it
#      prints a LENGTH-PER-LABEL report; read it in the log: if the labels
#      differ a lot in average length, expect Tier 2b to flag a length
#      confound, and re-pick the label scheme before trusting the probe,
#   2. extract ROOT-only activations (--max-depth 0; no decomposition or
#      generation -- the probes use only the root vector, so this is one
#      forward pass per question),
#   3. run the analysis tiers listed in TIERS (Tier 1 is trivial at
#      --max-depth 0, so no job lists it),
# writing every figure + summary to ${FIGS}/.
#
# Contract -- the sourcing script sets:
#   DATASET           one name that derives the rest: the builder is
#                     src/datasets/build_${DATASET}.py, questions land in
#                     data/questions_${DATASET}/${DATASET}.jsonl, the download
#                     hint is scripts/download_${DATASET}.sh, and log lines
#                     are tagged [run_${DATASET}]
#   RAW_JSON          raw input file for the build step
#   RAW_JSON_VAR      NAME of its submit-time override var (error hint only)
#   LABEL_SCHEME      builder --label-scheme
#   OUT / FIGS        activations+metadata dir / figures+summaries dir -- the
#                     sourcing script namespaces these by MODEL_SLUG (e.g.
#                     outputs/<slug>/run_<dataset>) so several models coexist;
#                     MODEL (the HF id) is passed to `main.py --model`. Both come
#                     from scripts/model_env.sh (sourced below + by the launcher).
#   TIERS             analyze.py tiers to run, e.g. "2 2b 2g" -- list 2g only
#                     for label schemes with >=3 ordered classes
#   BUILD_EXTRA_ARGS  array of extra builder flags (e.g. 2wiki's hop bounds);
#                     () when the builder needs none
# and may leave to the shared defaults below:
#   PER_CLASS, PERMS
# ---------------------------------------------------------------------------

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "ERROR: _run_dataset_common.sh is sourced by the run_<dataset>.sh jobs, not run directly." >&2
    exit 1
fi

for _v in DATASET RAW_JSON RAW_JSON_VAR LABEL_SCHEME OUT FIGS TIERS; do
    if [ -z "${!_v:-}" ]; then
        echo "ERROR: _run_dataset_common.sh: \$${_v} must be set by the sourcing run_<dataset>.sh" >&2
        exit 1
    fi
done

PER_CLASS="${PER_CLASS:-300}"   # balanced questions per class
PERMS="${PERMS:-200}"           # Tier 2b label-permutation shuffles
TAG="[run_${DATASET}]"

# Cluster helpers (EXIT/INT/TERM cleanup trap) + env (torch/transformers/numpy/
# matplotlib, HF_HOME). fathom uses plain HF transformers BY DESIGN (no vLLM).
source ../scripts/helper.sh
source ../scripts/set_up_env.sh
# MODEL / MODEL_SLUG (the sourcing run_<dataset>.sh already set MODEL_SLUG to
# namespace OUT/FIGS; re-source so MODEL is guaranteed set for --model below).
source ../scripts/model_env.sh

# Hop to the repo root so the launchers and CLI paths resolve as documented.
cd ..

# 1) Build the labeled jsonl (balanced classes; LENGTH-PER-LABEL report).
python -m "src.datasets.build_${DATASET}" \
    --input "${RAW_JSON}" \
    --output "data/questions_${DATASET}/${DATASET}.jsonl" \
    --label-scheme "${LABEL_SCHEME}" \
    "${BUILD_EXTRA_ARGS[@]}" \
    --per-class "${PER_CLASS}" \
    --seed 0
build_status=$?
if [ ${build_status} -ne 0 ]; then
    echo "${TAG} dataset build failed (status ${build_status}) -- is ${RAW_JSON_VAR} correct? (${RAW_JSON})"
    echo "${TAG} (download it first on a login node: bash scripts/download_${DATASET}.sh)"
    exit ${build_status}
fi

# 2) Extract ROOT activations only (no decomposition, no generation).
#    Re-running the same --output resumes (skips existing activations/<id>.npz).
#    --model overrides config.yaml's model_name so MODEL picks the run's LM.
python main.py \
    --input "data/questions_${DATASET}" \
    --output "${OUT}" \
    --config config.yaml \
    --model "${MODEL}" \
    --max-depth 0 \
    --device cuda
status=$?
echo "${TAG} main.py exited with status ${status}"

# 3) The TIERS analyses in this same job. A plotting hiccup must NOT fail the
#    (expensive) extraction, so per-tier statuses are logged but the job still
#    exits with main.py's status.
run_tier() {
    local tier="$1"; shift
    python analyze.py --output "${OUT}" --tier "${tier}" --plots-dir "${FIGS}" "$@"
    echo "${TAG}   tier ${tier} exited with status $?"
}

if [ ${status} -eq 0 ]; then
    for tier in ${TIERS}; do
        case "${tier}" in
            2)  echo "${TAG} Tier 2 probes -> ${FIGS}/ ..."
                run_tier 2 ;;
            2b) echo "${TAG} Tier 2b controls (length + multi-covariate + ${PERMS}-shuffle null) ..."
                run_tier 2b --perms "${PERMS}" ;;
            2g) echo "${TAG} Tier 2g generalization across depth (train extremes, test middle) ..."
                run_tier 2g ;;
            *)  echo "${TAG} Tier ${tier} ..."
                run_tier "${tier}" ;;
        esac
    done
fi

exit ${status}
