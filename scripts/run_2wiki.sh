#!/bin/bash
#SBATCH -J fathom-2wiki
#SBATCH --mem=64GB
#SBATCH -t 0-02:59:00
#SBATCH --mail-type=NONE
#SBATCH --chdir=../src/
#SBATCH -o ../outputs/slurm_logs/fathom_2wiki_%j.out
#SBATCH --account=def-yassine
#SBATCH --gpus-per-node=h100:1

# ---------------------------------------------------------------------------
# fathom -- THE single end-to-end 2WikiMultihopQA job: build the labeled
# jsonl -> extract ROOT-only activations -> Tier 2 probes + 2b controls,
# every figure + summary to figures/run_2wiki/. The shared mechanics live in
# scripts/_run_dataset_common.sh (see its header for the variable contract);
# this file is the SBATCH header plus 2Wiki's knobs.
#
# Difficulty signal = HOP COUNT (length of the gold evidence chain): a sharp,
# FATHOM-aligned operationalization of "reasoning depth". The default
# hops_binary scheme gives a clean easy(<=2) vs hard(>=4) contrast; Tier 2t
# (run_transfer.sh) maps it onto MuSiQue's graded 2hop/3hop/4hop automatically
# (labels are ordered per run; low<->low, high<->high).
#
# Submit from scripts/:   cd scripts && sbatch run_2wiki.sh
# (or let scripts/run_pipeline.sh submit it in parallel with the other two,
#  followed by the dependent run_transfer.sh)
#
# PREREQ (one-time, on a LOGIN node -- compute nodes have no internet):
#   bash scripts/download_2wiki.sh            # -> data/raw/2wiki/dev.json
#
# Tunables -- override at submit time, e.g.:
#   PER_CLASS=500 PERMS=100 sbatch run_2wiki.sh
#   LABEL_SCHEME=hops_multiclass sbatch run_2wiki.sh   # '2hop'/'4hop' labels instead
# (WIKI_OUT / WIKI_FIGS are deliberately NOT named OUT / FIGS: run_pipeline.sh
#  propagates the environment to ALL the GPU jobs, and shared names would let
#  one override point two datasets' runs at the SAME directory.)
# ---------------------------------------------------------------------------
source ../scripts/model_env.sh                    # MODEL (--model) + MODEL_SLUG (path namespace)
DATASET=2wiki
RAW_JSON_VAR=WIKI_JSON
RAW_JSON="${WIKI_JSON:-data/raw/2wiki/dev.json}"
LABEL_SCHEME="${LABEL_SCHEME:-hops_binary}"               # easy(<=2) / hard(>=4)
OUT="${WIKI_OUT:-outputs/${MODEL_SLUG}/run_2wiki}"        # activations + metadata (per-model)
FIGS="${WIKI_FIGS:-figures/${MODEL_SLUG}/run_2wiki}"      # all figures + summaries (per-model)
BUILD_EXTRA_ARGS=(--easy-hops 2 --hard-hops 4)    # hops_binary class bounds
TIERS="2 2b"   # NO 2g: hops_binary has 2 classes; 2g needs >=3 ordered classes

source ../scripts/_run_dataset_common.sh
