# Probing Compositional Question Difficulty in Language Model Representations

Omar Abdelhedi, Omid Reza Heidari, Sahar Chaari, Diaa Azzam, Manar Hamed, Xiang Chen Zhu, Yassine Yaakoubi.
*IEEE AIDIST 2026.*

This repository holds the code for the paper. We ask whether a language
model encodes how hard a multi-hop question is in its hidden state of the
question itself, before it produces any answer. For each question we run one
forward pass, keep the hidden state of every layer, and fit a linear probe per
layer. We then test the signal against surface features, a label-permutation
null, a held-out middle difficulty class, and transfer to other benchmarks.

## Repository layout

```
main.py              extract per-layer hidden states (GPU)
analyze.py           within-dataset analyses: Tiers 2, 2b, 2g (CPU)
transfer.py          cross-dataset transfer sweep (CPU)
config.yaml          extraction settings (model, dtype, pooling, layers, seed, batch size)
src/
  datasets/          build_{2wiki,hotpot,musique}.py: raw benchmark -> balanced labeled jsonl
  load_questions.py  pipeline.py  activations.py      extraction
  run_data.py        loading a finished run
  probe_core.py      the probe, CV, residualisation, bootstrap, ordinality math
  surface.py         the surface features
  probe.py           Tier 2:  decodability by layer
  probe_controls.py  Tier 2b: surface controls, permutation null, bootstrap CIs
  probe_generalization.py   Tier 2g: ordinality
  transfer.py        cross-dataset transfer
tests/
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer. Only `main.py` needs a GPU. The paper's runs used one
NVIDIA H100 per extraction, with the model in bfloat16. A 32B model needs about
65 GB of GPU memory; with less, the weights are sharded across several GPUs.
Everything else runs on CPU with numpy and matplotlib.

The paper's extraction ran with Python 3.11 and transformers 5.8.0 (each
run's `run_config.json` records its version). The test suite passes with
torch 2.15, transformers 5.24 and numpy 2.5.

## Data

The benchmarks are not redistributed here. Download them into `data/raw/`:

```bash
mkdir -p data/raw/2wiki data/raw/hotpot data/raw/musique

# 2WikiMultiHopQA, dev split (https://github.com/Alab-NII/2wikimultihop)
curl -L -o /tmp/2wiki.zip "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1"
unzip -j /tmp/2wiki.zip '*dev.json' -d data/raw/2wiki/

# HotpotQA, TRAIN split (https://hotpotqa.github.io)
curl -L -o data/raw/hotpot/hotpot_train_v1.1.json \
    http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json

# MuSiQue, answerable dev split (https://github.com/StonyBrookNLP/musique):
# get musique_v1.0.zip with the upstream download_data.sh, then
unzip -j musique_v1.0.zip '*musique_ans_v1.0_dev.jsonl' -d data/raw/musique/
```

HotpotQA must be the **train** split: its `level` annotation is all `hard` on
dev and absent on test. If a link has moved, take the file from the upstream
repository; the 2Wiki archive you need is `data_ids_april7.zip`, not the
paragraph dump listed next to it.

| Benchmark | Label used | How it is derived | Licence |
|---|---|---|---|
| 2WikiMultiHopQA | easy / hard | evidence-chain length: ≤ 2 items easy, ≥ 4 hard, 3 dropped | Apache-2.0 |
| HotpotQA | easy / medium / hard | the native, human-annotated `level` | CC BY-SA 4.0 |
| MuSiQue | 2hop / 3hop / 4hop | `len(question_decomposition)` | CC BY 4.0 |

Each builder samples 300 questions per class (seed 0), so a run has 600
questions (2Wiki) or 900 (HotpotQA, MuSiQue). The question sets you build are
derived from the benchmarks and fall under their licences, including
HotpotQA's share-alike clause. `data/` and `outputs/` are git-ignored.

## Running the experiments

The paper uses three models; the slugs name their output folders:

| Slug | HuggingFace model | Hidden states (incl. embeddings) | Hidden size |
|---|---|---|---|
| `qwen32b` | `Qwen/Qwen2.5-32B-Instruct` | 65 | 5120 |
| `mistral24b` | `mistralai/Mistral-Small-24B-Instruct-2501` | 41 | 5120 |
| `gemma27b` | `google/gemma-2-27b-it` | 47 | 4608 |

**1. Build the question sets** (once; model-independent):

```bash
python -m src.datasets.build_2wiki   --input data/raw/2wiki/dev.json \
    --output data/questions_2wiki/2wiki.jsonl
python -m src.datasets.build_hotpot  --input data/raw/hotpot/hotpot_train_v1.1.json \
    --output data/questions_hotpot/hotpot.jsonl
python -m src.datasets.build_musique --input data/raw/musique/musique_ans_v1.0_dev.jsonl \
    --output data/questions_musique/musique.jsonl
```

The defaults (`--per-class 300 --seed 0`) are the paper's settings.

**2. Extract activations** (GPU), for one model:

```bash
MODEL=Qwen/Qwen2.5-32B-Instruct
SLUG=qwen32b
for DS in 2wiki hotpot musique; do
  python main.py --input data/questions_$DS --output outputs/$SLUG/run_$DS --model $MODEL
done
```

Each question is tokenised as raw text (no chat template) and passed through
the model once. The run directory gets `activations/<id>.npz` (one float32
array `[n_layers, hidden_size]` under the key `root`), `metadata.jsonl` and
`run_config.json`. Re-running the same command resumes an interrupted run.

**3. Within-dataset analyses** (CPU):

```bash
for DS in 2wiki hotpot musique; do
  python analyze.py --output outputs/$SLUG/run_$DS --tier all --plots-dir figures/$SLUG/run_$DS
done
```

Tier 2g needs three ordered classes, so it skips 2Wiki. The defaults (5
folds, seed 0, 200 permutations, 2000 bootstrap resamples) are the paper's
settings. Tier 2b's 200-permutation null refits the probe 200 times at every
layer and is by far the slowest step.

**4. Cross-dataset transfer** (CPU), once all three runs of a model exist:

```bash
python transfer.py --runs outputs/$SLUG/run_2wiki outputs/$SLUG/run_hotpot outputs/$SLUG/run_musique \
    --plots-root figures/$SLUG
```

Repeat steps 2–4 for `mistral24b` and `gemma27b`.

## What each step writes

Figures and summaries go to the `--plots-dir` / `--plots-root` you pass
(`figures/` above, which is git-ignored).

| Paper | Output file |
|---|---|
| Fig. 1 | `run_<dataset>/tier2_probe_auc_by_layer.png` |
| Fig. 2 | `run_<dataset>/tier2b_permutation_null_by_layer.png` |
| Fig. 3 | `run_<dataset>/tier2b_multicovariate_control_by_layer.png` |
| Fig. 4 | `run_<dataset>/tier2g_generalization_by_layer.png` |
| Fig. 5 | `run_<dataset>/tier2_diff_projection_by_layer.png` |
| Fig. 6 | top row of `transfer_sweep/transfer_<a>_<b>--<test>/fused_transfer_report.png` |

| Table column | JSON field |
|---|---|
| I: Layer, Probe AUC | `tier2_summary.json` → `probe.best_layer_index`, `probe.best_auc` |
| I: Surf. only | `tier2b_summary.json` → `surface_only_auc` |
| I: Surf. resid. | `tier2b_summary.json` → `bootstrap.surface_residualized.auc` |
| I: ± (both AUC columns) | `tier2b_summary.json` → `bootstrap.raw.half_width`, `bootstrap.surface_residualized.half_width` |
| I: Null95 | `tier2b_summary.json` → maximum over layers of `null_p95` |
| II: AUC, R (raw) | `fused_transfer_summary.json` → `direction.mean_transfer_auc_informative`, `direction.retention` |
| II: AUC, R (resid.) | `fused_transfer_residual_summary.json` → same fields |
| Ordinality (Sec. IV-D) | `tier2g_summary.json` → `mean_position_informative` |

Tier 2 also writes three supplementary plots that the paper does not use:
centroid separation, a PCA scatter, and activation norm by layer. Tier 2b
writes a length-only control. The transfer report adds per-dimension bar
plots under Fig. 6's two panels (`--report-level summary` drops them).

## Method details that affect the numbers

- **Pooling:** the hidden state of the last non-padding token, at every layer including the embeddings.
- **Probe:** diagonal LDA. Each feature is standardised with the training fold's mean and standard deviation, and the direction is the difference of class means. It has no regularisation parameter.
- **Evaluation:** stratified 5-fold CV, seed 0. AUC is computed once over the pooled out-of-fold scores. With three classes it is the macro one-vs-rest AUC.
- **Label order:** each run's labels are ordered easiest to hardest (easy < medium < hard; otherwise alphabetical, so 2hop < 3hop < 4hop). Transfer matches lowest to lowest and highest to highest.
- **Surface features** (`src/surface.py`): character length, word count, capitalised non-initial words, words containing a digit, the number of `,` and `?`, and the tokenised length. They are regressed out of every activation feature within each training fold.
- **Informative layers:** layers whose relevant AUC is above 0.70. Retention and middle-class positions are averaged over these.
- **Bootstrap:** the out-of-fold scores are held fixed and the questions resampled 2000 times. The CI is the 2.5th–97.5th percentile of the peak-layer AUC.

## Notes

- **Tokenised length needs the tokenizer.** Tier 2b and the transfer load the run's tokenizer (no weights) to compute it. If that fails (offline, for example), the feature is dropped with a printed note. The summaries' `covariates` field shows which features were used.
- **Re-extraction is not bit-identical** across GPUs, kernels and library versions. The analyses are deterministic given the activations.
- **Activations are large.** Stored as float32, the nine runs take about 6.8 GiB (Qwen 2.98, Mistral 1.88, Gemma 1.94 GiB). `outputs/` is git-ignored.

## Tests

```bash
python -m pytest tests
```

The analysis tests need only numpy and matplotlib. The extraction tests build
a tiny random model locally and run when torch and transformers are installed;
otherwise they skip.

## Citation

```bibtex
@inproceedings{abdelhedi2026probing,
  title     = {Probing Compositional Question Difficulty in Language Model Representations},
  author    = {Abdelhedi, Omar and Heidari, Omid Reza and Chaari, Sahar and
               Azzam, Diaa and Hamed, Manar and Zhu, Xiang Chen and Yaakoubi, Yassine},
  booktitle = {IEEE International Conference on Artificial Intelligence,
               Data Science and Innovative Technologies (AIDIST)},
  year      = {2026}
}
```

See also `CITATION.cff`.

## Licence

The code is released under Apache-2.0 (see `LICENSE`). The benchmarks keep
their own licences, listed in [Data](#data).
