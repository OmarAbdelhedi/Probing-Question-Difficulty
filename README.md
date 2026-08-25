# fathom

**fathom** extracts and stores LLM hidden-state activations for labeled QA
questions across **recursive, multi-level decomposition**.

Each original question is the **root** of a decomposition tree. The target model
breaks it into simpler sub-questions; any sub-question that is still complex gets
decomposed again into its own children, and so on. Hard questions grow deep
branches; simple ones stay shallow and stop early. We then run forward passes
over **every node** in every tree and store its per-layer pooled hidden state.

This stage of the project produces **trees + activations only**. There is no
answer generation and no scoring here — decomposition is the *only* generation
step, and the only similarity computation is the cosine self/diff check inside
the sanity script.

A single HuggingFace causal LM (default `Qwen/Qwen2.5-32B-Instruct`) is loaded
**once** and used for **both** roles: decomposition (`model.generate`) and
activation extraction (forward passes). Same model, same layers, same pooling
everywhere — this is validity-critical.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Paths vs. knobs](#paths-vs-knobs-cli-args-vs-configyaml)
- [Configuration (`config.yaml`)](#configuration-configyaml)
- [Input format](#input-format)
- [Output layout](#output-layout)
- [Decomposition and recursion](#decomposition-and-recursion)
- [Swapping the model / device / pooling / layers](#swapping-the-model--device--pooling--layers)
- [The decomposer seam](#the-decomposer-seam)
- [Analysis and plots](#analysis-and-plots)
- [Sanity checks](#sanity-checks)
- [Tests](#tests)

---

## Installation

> **Note:** this development environment has **no Python on `PATH`**. Install and
> run everything inside your own Python environment.

```bash
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
```

Dependencies are minimal: `torch`, `transformers`, `accelerate`, `numpy`,
`pyyaml`. (`accelerate` enables streaming a large model straight onto the GPU and
sharding it across GPUs via `device_map="auto"`.)
(vLLM is intentionally **not** used — we need token-aligned per-layer hidden
states for an offline batch job, and plain `transformers` avoids
prefix-caching / batch-misalignment pitfalls.)

---

## Quick start

Run the full pipeline over a directory of labeled questions:

```bash
python main.py --input data/questions --output outputs/run1
```

Smoke test on the first 2 questions:

```bash
python main.py --input data/questions --output outputs/smoke --limit 2
```

`main.py` (at the project root) is a **thin launcher**: it parses CLI args,
loads `config.yaml`, and calls `src.pipeline.run`. All real logic lives in the
importable `src/` package, with absolute imports rooted at `src` (e.g.
`from src.decompose import build_tree`), so `python main.py ...` works directly
without `python -m`.

### CLI flags

| Flag           | Required | Default          | Meaning                                                            |
| -------------- | -------- | ---------------- | ------------------------------------------------------------------ |
| `--input`      | **yes**  | —                | Directory of labeled questions (`*.jsonl`).                        |
| `--output`     | **yes**  | —                | Output directory (created, incl. an `activations/` subdir).       |
| `--config`     | no       | `./config.yaml`  | Experiment-parameter YAML file.                                    |
| `--limit N`    | no       | all              | Process only the first `N` questions.                             |
| `--max-depth K`| no       | from config      | Override `max_depth` from the config.                             |
| `--device`     | no       | from config      | `cpu` or `cuda`; overrides config `device`.                       |
| `--model`      | no       | from config      | HF model id; overrides config `model_name` (used for **both** decomposition and extraction). |

`--input` is validated to exist and `--output` is validated to be writable
**before** the model is loaded, so bad paths fail fast with a clear message. If
`cuda` is requested but unavailable, fathom falls back to `cpu` with a warning.

---

## Paths vs. knobs (CLI args vs. `config.yaml`)

A deliberate separation:

- **Paths are CLI arguments.** Where to read questions from (`--input`) and where
  to write artifacts (`--output`) are passed on the command line. `config.yaml`
  contains **no paths**.
- **Experiment knobs live in `config.yaml`.** The model, dtype, pooling, layers,
  recursion limits, batch size, seed, etc. are config values so a run is fully
  reproducible from a single file.

`--max-depth`, `--device`, and `--model` are the knobs that can also be overridden
on the command line, for convenience when sweeping models or depth; everything else
is config-only.

---

## Configuration (`config.yaml`)

`config.yaml` holds **experiment parameters only** (no paths). Each key:

| Key                 | Default                        | Meaning                                                                                          |
| ------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------ |
| `model_name`        | `Qwen/Qwen2.5-32B-Instruct`    | HF model id used for **both** decomposition and activation extraction; overridable per-run with `--model` (the SLURM scripts pass `MODEL=<hf-id>` — see "Running multiple models" below). |
| `dtype`             | `bfloat16`                     | Compute dtype: `float16` / `bfloat16` / `float32`. bf16 is Qwen2.5's native dtype (best for a 32B on H100); fp16 auto-falls back to fp32 on CPU. |
| `device`            | `cuda`                         | Default device (`cpu` / `cuda`); overridable with `--device`.                                    |
| `pooling`           | `last_token`                   | How per-token hidden states collapse to one vector: `last_token` or `mean` (always mask-aware).  |
| `layers`            | `all`                          | Which hidden-state layers to store: `all`, or an explicit list of indices.                       |
| `max_depth`         | `3`                            | Hard cap on decomposition depth (root is depth 0); overridable with `--max-depth`.               |
| `node_budget`       | `64`                           | Per-question circuit breaker: max nodes before expansion stops.                                  |
| `decompose_context` | `isolated`                     | `isolated` decomposes each child **without** showing its parent; `with_parent` includes it.      |
| `seed`              | `0`                            | Seed for `torch` / `numpy` / `random` for deterministic runs.                                    |
| `batch_size`        | `16`                           | Batch size for the activation-extraction forward passes.                                          |

**Pooling never includes padding.** `last_token` takes the last *real* token per
sequence (padding-side agnostic, via the attention mask); `mean` is a
mask-weighted average over real tokens only.

**Layers.** `hidden_states` has length `num_transformer_layers + 1` (index `0` is
the embedding layer). `all` stores every layer; a list like `[0, 12, 24]` stores
only those. For provenance, the resolved `layers` written into `metadata.jsonl`
and `run_config.json` is the **config value** (e.g. `"all"`), not the expanded
index list.

**Model size / memory.** The default `Qwen/Qwen2.5-32B-Instruct` needs a GPU: in
`bfloat16` it is roughly 62 GB of weights, which fits on a single 80 GB H100. On
GPU the model is loaded with `device_map="auto"`, so if one GPU is not enough the
weights are automatically **sharded across all visible GPUs** (request more with
`--gpus-per-node=h100:2` in `scripts/run_musique.sh`). For a quick CPU / login-node
smoke test, temporarily set a small model such as `Qwen/Qwen2.5-1.5B-Instruct`.

---

## Input format

`--input` points to a **directory** containing one or more `*.jsonl` files. Each
line is a single JSON object:

```json
{"id": "q001", "question": "Why is the sky blue?", "label": "science"}
```

Requirements (enforced by `src/load_questions.py`):

- at least one `*.jsonl` file must exist in the directory;
- every line must be valid JSON with all three keys `id`, `question`, `label`;
- `id` must be non-empty and **unique across all files**;
- questions are returned in `(file, line)` order.

Malformed lines, empty files, or duplicate ids raise an error naming the offending
file and line.

> **Switching the input format** (e.g. to CSV, or a directory-of-files layout)
> touches **only `src/load_questions.py`** — no other module reads raw inputs.

---

## Output layout

Everything is written under `--output`:

```
<output>/
├── activations/
│   ├── q001.npz
│   ├── q002.npz
│   └── ...
├── metadata.jsonl
└── run_config.json
```

- **`activations/<id>.npz`** — one file per question. Because each tree is ragged,
  the `.npz` is **node-keyed**: one array per node, keyed by its node path
  (`root`, `root.0`, `root.0.1`, ...). Each array has shape
  `[n_selected_layers, hidden_dim]`. The root node is included naturally.
- **`metadata.jsonl`** — one record per **question**:

  ```json
  {
    "id": "q001",
    "label": "science",
    "original_question": "Why is the sky blue?",
    "pooling": "last_token",
    "layers": "all",
    "max_depth": 3,
    "decompose_context": "isolated",
    "nodes": [
      {"id": "root", "parent_id": null, "depth": 0, "question": "...",
       "stop_reason": null, "parse_failed": false},
      {"id": "root.0", "parent_id": "root", "depth": 1, "question": "...",
       "stop_reason": "atomic", "parse_failed": false}
    ]
  }
  ```

  The `nodes` list is a pre-order flatten of the tree. `parent_id` is the node id
  with the trailing `.<n>` stripped; the root has `parent_id = null`.

- **`run_config.json`** — full reproducibility: resolved `model_name`, `dtype`,
  `layers`, `pooling`, `max_depth`, `node_budget`, `decompose_context`, `seed`,
  `input_dir`, `output_dir`, `transformers_version`, and a `timestamp`.

### Node path scheme

Ids encode the tree position: the root is `root`; its `i`-th child is `root.i`;
that child's `j`-th child is `root.i.j`, and so on. The `.npz` keys and the
`metadata.jsonl` node ids use this same scheme, so an array and its metadata
record line up by id.

### Resumability

The pipeline is a **two-pass, resumable** job:

1. **Pass 1 — build trees** (the *only* step that calls `model.generate`).
2. **Pass 2 — extract activations** (forward passes only): flatten every tree,
   then batch-extract activations for all node texts at once.

A question whose `activations/<id>.npz` already exists is **skipped**. The `.npz`
is written **before** its `metadata.jsonl` line, so an interrupted run can be
resumed safely.

At the end, fathom prints: questions processed, total nodes, and a per
`stop_reason` breakdown.

---

## Decomposition and recursion

The same decomposition prompt (`DECOMPOSE_PROMPT` in `src/decompose.py`) is
applied **identically at every node** (root and sub-questions alike). The model
either returns `[]` (the question is atomic — answerable directly) or a JSON array
of strictly-narrower, independently-answerable sub-questions. Output is parsed
robustly: the first balanced JSON array is extracted; any parse failure is treated
as atomic (`[]`) and flagged with `parse_failed=true` rather than crashing.

Recursion has **two independent stop conditions** (both implemented; neither
collapses into the other):

- **`max_depth`** — hard cap: at `depth >= max_depth`, stop with no children.
- **`atomic`** — adaptive: `decompose()` returned `[]` (or all sub-questions were
  restatements of the parent), so the node is a leaf. Easy branches stop early —
  fathom does **not** force uniform depth.

Plus guards:

- **`budget`** — a per-question node-count circuit breaker (`node_budget`).
- a child whose normalized text equals its parent's is dropped (no rephrasing).

Every leaf ends with a non-null `stop_reason` in `{atomic, max_depth, budget}`.

---

## Swapping the model / device / pooling / layers

- **Model** — set `model_name` in `config.yaml` to any HF causal LM id. The same
  model is used for decomposition and activations.
- **`max_depth`** — set in `config.yaml`, or override per run with `--max-depth K`.
- **Device** — set `device` in `config.yaml`, or override with `--device cpu|cuda`.
  (`cuda` auto-falls back to `cpu` when unavailable; `float16` auto-falls back to
  `float32` on CPU.)
- **Pooling** — set `pooling` to `last_token` (default) or `mean`.
- **Layers** — set `layers` to `all` (default) or an explicit list of indices.

---

## The decomposer seam

Decomposition is intentionally pluggable. `make_target_decomposer(model,
tokenizer, ...)` in `src/decompose.py` returns a `decompose_fn(question,
parent_question=None) -> (subquestions, parse_failed)` that runs on the target
model.

To use an **API decomposer** later, write another factory that returns a callable
with the **same** `(question, parent_question) -> (list, bool)` signature and pass
it to `build_tree`. Only the decomposition step moves off-model — **activations
always stay on the target model**, preserving experimental validity.

---

## Analysis and plots

After a run, `analyze.py` (a thin launcher mirroring `main.py`) turns the run's
artifacts into figures. It has five tiers, chosen with `--tier {1,2,2b,2g,2t,all}`
(default `all` = the four single-run tiers; the cross-run Tier 2t additionally
needs `--test-output` and is included in `all` when that flag is given);
**all are pure numpy + matplotlib — no torch / sklearn** (Tier 2b
optionally loads a weight-free tokenizer) — so they run on a login node or a laptop
straight from the run output. All figures + summaries go to a **`figures/` folder
at the project root** (auto-created if missing; override with `--plots-dir`).

```bash
python analyze.py --output outputs/run1            # Tiers 1, 2, 2b, 2g
python analyze.py --output outputs/run1 --tier 1   # metadata sanity plots only
python analyze.py --output outputs/run1 --tier 2   # activation probes only
python analyze.py --output outputs/run1 --tier 2b  # probe controls only
python analyze.py --output outputs/run1 --tier 2g  # generalization across depth
python analyze.py --output outputs/run_musique --tier 2t \
    --test-output outputs/run_2wiki \
    --plots-dir figures/transfer_musique_2wiki     # cross-dataset transfer
```

| Flag                     | Tier          | Default                        | Meaning                                                                  |
| ------------------------ | ------------- | ------------------------------ | ------------------------------------------------------------------------ |
| `--output`               | all           | — (**required**)               | The run dir passed to `main.py --output` (Tier 2t: the TRAIN run).       |
| `--tier`                 | all           | `all`                          | `1` sanity, `2` probes, `2b` controls, `2g` depth-generalization, `2t` cross-dataset transfer. |
| `--metadata`             | all           | `<output>/metadata.jsonl`      | Override the metadata path.                                              |
| `--plots-dir`            | all           | `<repo_root>/figures`          | Override where figures + summaries are written.                          |
| `--activations-dir`      | 2, 2b, 2g, 2t | `<output>/activations`         | Dir of `<id>.npz` activations.                                           |
| `--test-output`          | 2t            | —                              | The SECOND finished run to transfer onto (same model/pooling/layers).    |
| `--test-metadata`        | 2t            | `<test-output>/metadata.jsonl` | Override the second run's metadata path.                                 |
| `--test-activations-dir` | 2t            | `<test-output>/activations`    | Override the second run's activations dir.                               |
| `--folds`                | 2, 2b, 2g, 2t | `5`                            | Probe CV folds (clamped to the smallest class count).                    |
| `--seed`                 | 2, 2b, 2g, 2t | `0`                            | RNG seed for CV folds / PCA subsampling / permutations.                  |
| `--perms`                | 2b            | `200`                          | Label-permutation shuffles for the null.                                 |

`scripts/run_musique.sh` (and its `run_2wiki.sh` / `run_hotpot.sh` siblings) runs the
whole pipeline (build → extract → Tier 2 + 2b + 2g) as a single SLURM job; once ≥3
runs exist, `scripts/run_transfer.sh` runs the **fused 2-train/1-test** transfer sweep
over them as a CPU-only job (a thin wrapper around `python transfer_suite.py fused`):
for every way to hold one run out, it trains the depth axis on the **fused pool of the
other two** and tests on the held-out run (default runs: MuSiQue, 2Wiki, HotpotQA → 3
combinations; with fewer than 3 ready it soft-skips). **`scripts/run_pipeline.sh`
(login node) chains it all from a clean checkout**: it downloads MuSiQue + 2Wiki +
HotpotQA if missing (`scripts/download_*.sh` — downloads need a login node; compute
nodes have no internet), submits `run_musique.sh`, `run_2wiki.sh` and `run_hotpot.sh`
to run in parallel, then submits `run_transfer.sh` with an `afterok` dependency on all
three (each fused combination needs all three runs; auto-cancelled if any fails, so it
never hangs pending). Labels are free-form:
nothing hardcodes `easy`/`hard`; known hardness words are ordered intuitively and
anything else sorts alphabetically.

**Running multiple models.** Pass `MODEL=<hf-id>` to any run script (or to
`run_pipeline.sh`) to override `config.yaml`'s `model_name` for that run: it is
forwarded to `main.py --model` (so the same model does decomposition **and**
extraction) **and** namespaces every artifact by a filesystem-safe slug —
`outputs/<slug>/run_<dataset>/` and `figures/<slug>/run_<dataset>/` (plus
`figures/<slug>/transfer_sweep/`) — so several models' outputs and figures coexist
without clobbering. The slug defaults to the HF id with `/`→`_`
(e.g. `Qwen_Qwen2.5-32B-Instruct`); set `MODEL_SLUG=<short>` to shorten it. Both
are defined once in `scripts/model_env.sh`. Example:
`MODEL=meta-llama/Llama-3.1-8B-Instruct bash scripts/run_pipeline.sh`.

### Tier 1 — does decomposition track hardness? (metadata-only)

Confirm the cheap thing first: if hard questions don't grow deeper / larger trees
than easy ones, there's no point looking at hidden states. Reads only
`metadata.jsonl`. Outputs:

- **`tier1_depth_nodecount_by_label.png`** — per-question **tree depth** (max node
  depth) and **node count** distributions, split by label (violins with the raw
  points overlaid, so small / discrete samples never mislead). Expect easy →
  shallow & small, hard → deep & large.
- **`tier1_stop_reason_composition_by_label.png`** — stacked bars of the
  `atomic` / `max_depth` / `budget` fractions over each label's **leaf** nodes.
  Easy questions should be dominated by root-`atomic`; hard ones branch and pick
  up `max_depth` / `budget` leaves.
- **`tier1_summary.json`** — the same numbers (per-label question counts, depth /
  node-count stats, leaf stop-reason breakdown, parse-failure counts) as text.

### Tier 2 — is hardness decodable from activations? (needs `activations/`)

Loads each question's **root-node** activation (`activations/<id>.npz` key
`root`, shape `[n_layers, hidden_dim]`) and asks whether the hardness label is
linearly decodable from the hidden state, and at which layer. The probe is a
**diagonal-LDA linear probe** (standardize per feature on the train fold, then
project onto the class-mean difference), scored by **cross-validated AUC over
pooled out-of-fold predictions** — regularization-free and stable in the
`hidden_dim ≫ n_questions` regime (swap in sklearn `LogisticRegression` if you
prefer). Skips gracefully if `activations/` is absent, if there are <2 labels
(probe / separation), or <3 questions (PCA). Outputs:

- **`tier2_probe_auc_by_layer.png`** — ⭐ cross-validated probe AUC (+ balanced
  accuracy) vs layer: *where* in the network hardness becomes linearly decodable.
- **`tier2_centroid_separation_by_layer.png`** — standardized class-centroid
  distance per layer; a CV-free companion to the probe curve.
- **`tier2_pca_scatter_by_layer.png`** — 2D *unsupervised* PCA of the **layer-to-layer
  activation diff** (the residual-stream update `act[L] − act[L−1]` each layer applies),
  at a few evenly-spaced transitions, colored by label (do the classes separate on their
  own once the large component carried unchanged across layers is removed?). Even on the
  diff the top PCs track total variance, mostly label-irrelevant, so classes can still
  look unseparated at the very layer the probe decodes them — which is what the next
  figure addresses.
- **`tier2_diff_projection_by_layer.png`** — the **supervised companion** to the PCA
  scatter: the same layers viewed along the **held-out class-difference axis** (x)
  instead of the top unsupervised PCs, with a residual PC on y for spread. Separation
  appears where the probe says it should; for ≥3 ordered classes the held-out middle
  class lands *between* the extremes (the Tier 2g interpolation, drawn). The x-axis is
  out-of-fold (extremes) / projected out-of-sample (middle), so it can't double-dip in
  the `hidden_dim ≫ n` regime the way an axis fit on the plotted points would.
- **`tier2_activation_norm_by_layer.png`** — mean L2 norm of the root vector per
  layer, split by label (guards against hardness being a raw-scale confound).
- **`tier2_summary.json`** — per-layer AUC / separation, the best layer, and config.

> **Scale note:** Tier 2 loads root vectors into one `[n_questions, n_layers,
> hidden_dim]` float32 array, and the PCA / diff-projection scatters subsample to
> 1500 points; both are fine for hundreds–to–low-thousands of questions. For much
> larger runs, probe per-layer in a stream.

### Tier 2b — is Tier 2 real, or a length confound? (needs `activations/`)

A high Tier-2 AUC is suspicious if "hard" questions are also *longer* / more
elaborate — the probe may just be reading surface form. Tier 2b runs three
controls on the same root activations:

- **`tier2b_length_control_by_layer.png`** — the raw probe AUC vs a
  **length-residualized** probe (question character-length regressed out of every
  feature *within each train fold*), plus a **length-only** reference line. If the
  residualized curve collapses toward the length-only line, it was mostly length.
- **`tier2b_multicovariate_control_by_layer.png`** — the stronger test:
  residualize out a whole basket of **surface features** at once — char length,
  word count, a named-entity proxy (capitalized-word count), numeric-token count,
  clause punctuation, and (if a tokenizer loads) exact token count — vs a
  **surface-only** reference. Signal that survives *this* is robustly beyond
  surface form, not merely beyond raw length.
- **`tier2b_permutation_null_by_layer.png`** — the real probe AUC vs a
  **shuffled-label null** band (default 200 permutations). The real curve must sit
  far above the null; if the null is already high, the probe is leaking.
- **`tier2b_summary.json`** — per-layer raw / length-residualized /
  surface-residualized / null AUCs, the length-only and surface-only AUCs, and a
  one-line verdict (also printed to the console).

The headline verdict is whether the **surface-residualized** curve stays well
above the null band: if yes, the decodability is real, surface-independent
structure; if it collapses to the null, it was surface form. (On the bundled toy
set it collapses; on MuSiQue hop-count it survives — see the analysis arc.) The
token-count covariate needs the run's tokenizer (`run_config.json` model id); it
is loaded weight-free and skipped gracefully if transformers/the model is absent.

### Tier 2g — generalization across depth (needs ≥3 ordered classes)

The strongest test that the signal is a genuine *ordinal depth* axis rather than
class-specific template/topic memorization. The probe is trained on only the
**extreme** classes (lowest vs highest hop count, e.g. `2hop` vs `4hop`); each
**held-out middle** class (`3hop` — never seen in training) is then projected onto
that learned axis.

- **`tier2g_generalization_by_layer.png`** — left: held-out low-vs-high AUC by
  layer (confirms the extreme axis is learned); right: the **interpolation
  fraction** of each middle class, `(μ_mid − μ_low)/(μ_high − μ_low)`, by layer.
  `0` = looks like the low class, `1` = like the high class; **`0 < pos < 1` means
  it interpolates** (ordinal), `≈ 0.5` is dead-centre.
- **`tier2g_summary.json`** — per-layer AUC and interpolation fractions, the best
  layer, and the mean position over informative layers, plus a one-line verdict.

If the middle class lands *between* the extremes it never trained on, the model
placed "3 hops" between "2" and "4" on its own — evidence of a real ordinal depth
representation. Requires a graded label scheme (e.g. MuSiQue `hops_multiclass`);
skips gracefully for binary labels.

### Tier 2t — does the axis transfer across datasets? (needs TWO runs)

The within-dataset tests can't rule out one thing: the "depth axis" might be
dataset-specific — keyed to MuSiQue's composed templates / topic mix — rather
than a general property of the model. Tier 2t trains the low-vs-high
diagonal-LDA axis on **all** of one run's extreme classes and evaluates it on a
**different** run (nothing in the test run is ever fit on), in **both
directions**, each against the test run's own within-dataset CV extreme AUC as
the ceiling. Label vocabularies don't need to match: each run's labels are
ordered, and the axis maps low↔low / high↔high, so MuSiQue's graded `2hop/3hop/4hop`,
2Wiki's `easy/hard`, and HotpotQA's `easy/medium/hard` all interoperate. HotpotQA's
`level` is a **human-annotated** difficulty axis (not a derived hop count), so the
all-pairs sweep tests something the other two can't on their own: whether
human-judged and structural difficulty share a single encoded direction. Both runs
**must** come from the same model / pooling / layers — verified against each run's
`run_config.json`; shape or provenance mismatches are hard errors.

- **`tier2t_transfer_by_layer.png`** — left: foreign-axis low-vs-high AUC on the
  held-out dataset by layer, both directions, with each within-dataset CV
  ceiling dashed. The headline metric is **retention**
  `= (transfer − 0.5) / (within − 0.5)` over informative layers: ≳0.7 means the
  axis is largely dataset-general; ≲0.3 means it is mostly dataset-specific.
  Right (when a test run has ≥3 ordered classes): the middle class's
  interpolation fraction **on the foreign axis** — Tier 2g's ordinality test,
  across datasets.
- **`tier2t_summary.json`** — per-direction transfer AUC / balanced accuracy
  (threshold from the train run — shift-sensitive, secondary) / retention /
  interpolation positions, the within-CV ceilings, and a per-direction verdict
  (also printed to the console).

The pipeline's `scripts/run_transfer.sh` no longer runs this pairwise tier — it now
runs the **fused 2-train/1-test** sweep (`transfer_suite.py fused`, writing
`figures/transfer_sweep/transfer_<train_a>_<train_b>--<test>/`). To produce the
pairwise `tier2t_*` artifacts above, run a single pair by hand (then optionally
consolidate them with `python transfer_suite.py pairwise`):

```bash
python analyze.py --output outputs/run_musique --tier 2t \
    --test-output outputs/run_2wiki --plots-dir figures/transfer_musique_2wiki
```

A synthetic no-GPU smoke test (two fabricated runs sharing a planted depth
direction) runs the whole tier end-to-end: `python -m src.probe_transfer`.

---

## Sanity checks

An end-to-end runnable check lives in `src/sanity_checks.py`. It runs the pipeline
on a couple of questions, then loads the artifacts back and:

- prints each question tree indented by depth (node question + `stop_reason`) and
  the shape of every stored array;
- asserts no node depth exceeds `max_depth`;
- asserts every leaf has a `stop_reason` in `{atomic, max_depth, budget}`;
- asserts no child text equals its parent's (the guard worked);
- asserts the flat node list reconstructs into a valid tree (every `parent_id`
  resolves; exactly one root per question);
- asserts `cosine(v, v) ≈ 1.0` and two different nodes give `cosine < 1.0`;
- asserts pooled length equals the model `hidden_size` and the layer count is as
  expected;
- asserts no `NaN`/`Inf`.

Run it directly (it inserts the project root onto `sys.path`, loads
`./config.yaml`, and defaults to input `data/questions` / output
`outputs/sanity`):

```bash
python src/sanity_checks.py
```

The smoke test is equivalent to:

```bash
python main.py --input data/questions --output outputs/smoke --limit 2
```

---

## Tests

A pytest suite in `tests/` covers everything that runs without a model: the
shared probe math in `src/probe_core.py` (each metric checked against a
brute-force reference implementation, not golden constants), JSON-array parsing
and `build_tree` (driven through stub decomposers — the seam means no model is
needed), the question/metadata loaders, mask-aware pooling and layer-spec
resolution, the dataset builders' label derivation and the shared
bucket→balance→write core, and every analysis tier **end-to-end** on fabricated
runs with a planted depth axis (Tier 2 probes, 2b controls, 2g generalization,
2t cross-dataset transfer — each asserting on the written summary JSON, e.g.
that the probe finds the planted layer and that a planted length confound
collapses under the Tier-2b control).

```bash
pip install pytest
pytest            # from the repo root; ~5 s, no GPU
```

The suite needs only numpy + matplotlib (+ torch for the pooling/decompose
tests). Tests that touch `torch` / `transformers` skip themselves when those
are missing, so the same suite passes on a laptop without the GPU stack and on
the cluster. Model-dependent end-to-end verification stays in
`src/sanity_checks.py` (see [Sanity checks](#sanity-checks)).
