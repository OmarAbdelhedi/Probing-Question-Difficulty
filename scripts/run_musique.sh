#!/bin/bash
#SBATCH -J fathom-musique
#SBATCH --mem=64GB
#SBATCH -t 0-02:59:00
#SBATCH --mail-type=NONE
#SBATCH --chdir=../src/
#SBATCH -o ../outputs/slurm_logs/fathom_musique_%j.out
#SBATCH --account=def-yassine
#SBATCH --gpus-per-node=h100:1

# ---------------------------------------------------------------------------
# fathom -- THE single end-to-end MuSiQue job: build the labeled jsonl ->
# extract ROOT-only activations -> Tier 2 probes + 2b controls + 2g
# generalization, every figure + summary to figures/run_musique/. The shared
# mechanics live in scripts/_run_dataset_common.sh (see its header for the
# variable contract); this file is the SBATCH header plus MuSiQue's knobs.
#
# Difficulty signal = COMPOSED HOP COUNT, graded 2hop/3hop/4hop: >=3 ordered
# classes, so Tier 2g (train extremes, test middle) applies. MuSiQue questions
# lengthen with hops -- that gap is the confound the Tier-2b length / surface
# controls adjudicate; read the build step's LENGTH-PER-LABEL report in the log.
#
# Submit from scripts/:   cd scripts && sbatch run_musique.sh
# (or let scripts/run_pipeline.sh submit it in parallel with the other two,
#  followed by the dependent run_transfer.sh)
#
# PREREQ (one-time, on a LOGIN node -- compute nodes have no internet):
#   bash scripts/download_musique.sh          # -> data/raw/musique/
#
# Tunables -- override at submit time, e.g.:
#   PER_CLASS=500 PERMS=100 sbatch run_musique.sh
#   LABEL_SCHEME=hops_binary sbatch run_musique.sh     # 2-vs-4 headline instead of graded
# ---------------------------------------------------------------------------
source ../scripts/model_env.sh                    # MODEL (--model) + MODEL_SLUG (path namespace)
DATASET=musique
RAW_JSON_VAR=MUSIQUE_JSON
RAW_JSON="${MUSIQUE_JSON:-data/raw/musique/musique_ans_v1.0_dev.jsonl}"
LABEL_SCHEME="${LABEL_SCHEME:-hops_multiclass}"          # graded 2hop/3hop/4hop (or hops_binary)
OUT="${OUT:-outputs/${MODEL_SLUG}/run_musique}"         # activations + metadata (per-model)
FIGS="${FIGS:-figures/${MODEL_SLUG}/run_musique}"       # all figures + summaries (per-model)
BUILD_EXTRA_ARGS=()                               # no extra builder flags
TIERS="2 2b 2g"                                   # graded labels -> 2g applies

source ../scripts/_run_dataset_common.sh
