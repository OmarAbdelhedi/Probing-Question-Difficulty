#!/bin/bash
#SBATCH -J fathom-hotpot
#SBATCH --mem=64GB
#SBATCH -t 0-02:59:00
#SBATCH --mail-type=NONE
#SBATCH --chdir=../src/
#SBATCH -o ../outputs/slurm_logs/fathom_hotpot_%j.out
#SBATCH --account=def-yassine
#SBATCH --gpus-per-node=h100:1

# ---------------------------------------------------------------------------
# fathom -- THE single end-to-end HotpotQA job: build the labeled jsonl ->
# extract ROOT-only activations -> Tier 2 probes + 2b controls + 2g
# generalization, every figure + summary to figures/run_hotpot/. The shared
# mechanics live in scripts/_run_dataset_common.sh (see its header for the
# variable contract); this file is the SBATCH header plus Hotpot's knobs.
#
# Difficulty signal = HotpotQA's NATIVE 'level' (easy/medium/hard): a HUMAN
# annotation, not a derived hop count. This is what HotpotQA contributes that
# MuSiQue/2Wiki cannot -- the same depth-encoding question, asked along a
# human-perceived axis. 'level' is an ORDERED 3-class vocabulary FATHOM already
# knows (src/run_data.py _HARDNESS_ORDER: easy < medium < hard), so it supports
# Tier 2g here AND transfers cleanly under run_transfer.sh (Tier 2t maps the
# extremes: easy<->low, hard<->high). Length tracks 'level' only weakly
# (difficulty is human-judged, not composed from hops), so a length confound
# is less likely here than for MuSiQue -- but Tier 2b still checks it.
#
# Sampled from the TRAIN split BY NECESSITY: 'level' is constant (all 'hard') on
# the dev/distractor and test sets, so ONLY train carries the easy/medium/hard
# contrast. Per-class balancing (PER_CLASS) keeps the GPU work small (~900 q at
# the default) despite train's 90k size.
#
# Submit from scripts/:   cd scripts && sbatch run_hotpot.sh
# (or let scripts/run_pipeline.sh submit it in parallel with the other two,
#  followed by the dependent run_transfer.sh)
#
# PREREQ (one-time, on a LOGIN node -- compute nodes have no internet):
#   module load gcc arrow && bash scripts/download_hotpot.sh   # -> data/raw/hotpot/hotpot_train_v1.1.json
#   (pyarrow ships via the Arrow module on Alliance, hence `module load gcc arrow`.)
#
# Tunables -- override at submit time, e.g.:
#   PER_CLASS=500 PERMS=100 sbatch run_hotpot.sh
#   LABEL_SCHEME=level_binary sbatch run_hotpot.sh    # easy-vs-hard headline (drops 'medium')
#   LABEL_SCHEME=type         sbatch run_hotpot.sh    # bridge-vs-comparison reasoning type
# (HOTPOT_OUT / HOTPOT_FIGS are deliberately NOT named OUT / FIGS: a pipeline
#  submitter propagates the environment to every GPU job, and shared names would
#  let one override point two datasets' runs at the SAME directory.)
# ---------------------------------------------------------------------------
source ../scripts/model_env.sh                    # MODEL (--model) + MODEL_SLUG (path namespace)
DATASET=hotpot
RAW_JSON_VAR=HOTPOT_JSON
RAW_JSON="${HOTPOT_JSON:-data/raw/hotpot/hotpot_train_v1.1.json}"  # train: 'level' varies only here
LABEL_SCHEME="${LABEL_SCHEME:-level}"                     # native easy/medium/hard (graded, ordered)
OUT="${HOTPOT_OUT:-outputs/${MODEL_SLUG}/run_hotpot}"    # activations + metadata (per-model)
FIGS="${HOTPOT_FIGS:-figures/${MODEL_SLUG}/run_hotpot}"  # all figures + summaries (per-model)
BUILD_EXTRA_ARGS=()                               # no extra builder flags
TIERS="2 2b 2g"   # 2g runs for the default 'level' (3 ordered classes), skips gracefully for level_binary/type

source ../scripts/_run_dataset_common.sh
