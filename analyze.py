"""FATHOM analysis entry point — THIN launcher.

Like ``main.py``, this module contains NO analysis logic. It only parses CLI
arguments and delegates to the analysis modules in ``src/``:

  * Tier 1 (``src.analyze.run``)  -- metadata-only sanity plots: does the model's
    decomposition behavior track the hardness label? (tree depth / node count /
    stop-reason composition). NO activations needed.
  * Tier 2 (``src.probe.run``)    -- activation probes: is hardness linearly
    decodable from the hidden state, and where? (probe AUC / centroid separation
    / PCA scatter / activation norm, all vs layer). Reads ``activations/``; skips
    gracefully if that dir is absent.
  * Tier 2b (``src.probe_controls.run``) -- probe CONTROLS: a length control
    (length-only + length-residualized probe), a multi-covariate surface control,
    and a label-permutation null, to test whether Tier 2's decodability is real or
    a surface-form confound.
  * Tier 2g (``src.probe_generalization.run``) -- generalization across depth:
    train the probe on the EXTREME classes and test whether the held-out middle
    class interpolates between them (a genuine ordinal-depth check). Needs >=3
    ordered classes (a graded label scheme); skips otherwise.
  * Tier 2t (``src.probe_transfer.run``) -- CROSS-DATASET transfer: train the
    low-vs-high depth axis on this run (``--output``) and test it on a SECOND
    finished run (``--test-output``) -- both directions, each against its
    within-dataset CV ceiling, plus cross-dataset middle-class interpolation.
    The strongest cheap evidence that the axis is dataset-general rather than
    template-specific. Both runs must share the model/pooling/layers.

Run it AFTER ``main.py``, pointing ``--output`` at the SAME run directory. Both
tiers are pure numpy + matplotlib (NO torch / transformers / sklearn), so this
runs on a laptop straight from the run artifacts. All outputs go to a
``figures/`` folder at the project root (created if missing). Absolute imports
are rooted at ``src`` so ``python analyze.py ...`` works directly without
``python -m``.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "FATHOM analysis: Tier-1 metadata sanity plots (decomposition vs "
            "hardness label) and/or Tier-2 activation probes (is hardness "
            "linearly decodable, and where?), from a finished run's artifacts."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "REQUIRED. The run directory passed to main.py --output. Tier 1 reads "
            "its metadata.jsonl; Tier 2 reads its activations/. Plots default to a "
            "figures/ folder at the project root (override with --plots-dir)."
        ),
    )
    parser.add_argument(
        "--tier",
        choices=["1", "2", "2b", "2g", "2t", "all"],
        default="all",
        help="Which tier(s) to run: 1=metadata sanity, 2=activation probes, "
             "2b=probe controls (length/surface + permutation null), "
             "2g=generalization across depth (needs >=3 ordered classes), "
             "2t=cross-dataset transfer (needs --test-output), "
             "all=every single-run tier (default; 2t also runs iff --test-output "
             "is given).",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to metadata.jsonl (default: <output>/metadata.jsonl).",
    )
    parser.add_argument(
        "--plots-dir",
        dest="plots_dir",
        default=None,
        help="Directory for the output plots + summaries (default: <repo_root>/figures).",
    )
    parser.add_argument(
        "--activations-dir",
        dest="activations_dir",
        default=None,
        help="Tiers 2/2b/2g (and 2t's train side): dir of <id>.npz activations "
             "(default: <output>/activations).",
    )
    parser.add_argument(
        "--test-output",
        dest="test_output",
        default=None,
        help="Tier 2t only: a SECOND finished run directory to transfer the probe "
             "onto (train on --output, test on this, and vice versa). Must share "
             "the model/pooling/layers with --output.",
    )
    parser.add_argument(
        "--test-metadata",
        dest="test_metadata",
        default=None,
        help="Tier 2t only: metadata.jsonl for --test-output "
             "(default: <test-output>/metadata.jsonl).",
    )
    parser.add_argument(
        "--test-activations-dir",
        dest="test_activations_dir",
        default=None,
        help="Tier 2t only: activations dir for --test-output "
             "(default: <test-output>/activations).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Tiers 2/2b/2g/2t: cross-validation folds for the probe (clamped to "
             "the smallest class count; default 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Tiers 2/2b/2g/2t: RNG seed for CV folds / PCA subsampling / "
             "permutations (default 0).",
    )
    parser.add_argument(
        "--perms",
        type=int,
        default=200,
        help="Tier 2b only: label-permutation shuffles for the null (default 200).",
    )
    args = parser.parse_args()

    # Import inside main so argument errors surface before the matplotlib/numpy
    # import chain (and so a Tier-1-only run never imports the probe module).
    if args.tier in ("1", "all"):
        from src.analyze import run as run_tier1
        run_tier1(
            output_dir=args.output,
            metadata_path=args.metadata,
            plots_dir=args.plots_dir,
        )
    if args.tier in ("2", "all"):
        from src.probe import run as run_tier2
        run_tier2(
            output_dir=args.output,
            metadata_path=args.metadata,
            activations_dir=args.activations_dir,
            plots_dir=args.plots_dir,
            folds=args.folds,
            seed=args.seed,
        )
    if args.tier in ("2b", "all"):
        from src.probe_controls import run as run_tier2b
        run_tier2b(
            output_dir=args.output,
            metadata_path=args.metadata,
            activations_dir=args.activations_dir,
            plots_dir=args.plots_dir,
            folds=args.folds,
            seed=args.seed,
            perms=args.perms,
        )
    if args.tier in ("2g", "all"):
        from src.probe_generalization import run as run_tier2g
        run_tier2g(
            output_dir=args.output,
            metadata_path=args.metadata,
            activations_dir=args.activations_dir,
            plots_dir=args.plots_dir,
            folds=args.folds,
            seed=args.seed,
        )
    if args.tier == "2t" or (args.tier == "all" and args.test_output):
        if not args.test_output:
            parser.error("--tier 2t requires --test-output (the second finished run "
                         "to transfer the probe onto)")
        from src.probe_transfer import run as run_tier2t
        run_tier2t(
            output_dir=args.output,
            test_output_dir=args.test_output,
            metadata_path=args.metadata,
            activations_dir=args.activations_dir,
            test_metadata_path=args.test_metadata,
            test_activations_dir=args.test_activations_dir,
            plots_dir=args.plots_dir,
            folds=args.folds,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
