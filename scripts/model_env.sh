# ---------------------------------------------------------------------------
# fathom -- shared MODEL identity for the run scripts. SOURCED (never executed):
# every per-dataset job, the transfer job, and the pipeline submitter source
# this so the model id and its filesystem-safe slug are derived in ONE place and
# cannot drift between them.
#
#   MODEL       HF causal LM id, passed to `python main.py --model` so the SAME
#               model does decomposition AND activation extraction. Overrides
#               config.yaml's model_name at submit time (config stays the default).
#   MODEL_SLUG  filesystem-safe label for MODEL (each '/' in an HF id becomes
#               '_'), used to NAMESPACE outputs/<slug>/run_* and figures/<slug>/...
#               so several models' results + figures coexist without clobbering.
#
# Override either at submit time, e.g.:
#   MODEL=meta-llama/Llama-3.1-8B-Instruct sbatch run_2wiki.sh
#   MODEL=meta-llama/Llama-3.1-8B-Instruct MODEL_SLUG=llama8b sbatch run_2wiki.sh
#
# Keep the default MODEL below in sync with config.yaml's model_name (it is the
# value used when MODEL is unset, so the two never disagree on the default).
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
MODEL_SLUG="${MODEL_SLUG:-${MODEL//\//_}}"
export MODEL MODEL_SLUG
