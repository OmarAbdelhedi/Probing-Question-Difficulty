"""Within-dataset analyses of one extracted run (CPU only).

    python analyze.py --output outputs/qwen32b/run_musique --tier all \\
        --plots-dir figures/qwen32b/run_musique

  --tier 2   decodability by layer: probe AUC, Fig. 1; projections, Fig. 5
  --tier 2b  surface controls, permutation null, bootstrap CIs (Figs. 2-3, Table I)
  --tier 2g  ordinality: held-out middle class (Fig. 4; needs >= 3 classes)
  --tier all all three (2g skips binary benchmarks)

Needs numpy and matplotlib only; Tier 2b additionally tries to load the run's
tokenizer (no weights) for the token-count surface feature.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe a run's activations for difficulty (Tiers 2, 2b, 2g)."
    )
    parser.add_argument("--output", required=True,
                        help="Run directory written by main.py (activations/ + metadata.jsonl).")
    parser.add_argument("--tier", choices=["2", "2b", "2g", "all"], default="all",
                        help="Which analysis to run (default: all).")
    parser.add_argument("--plots-dir", dest="plots_dir", default=None,
                        help="Where figures and summaries go (default: <output>/figures).")
    parser.add_argument("--folds", type=int, default=5,
                        help="Cross-validation folds, clamped to the smallest class (default 5).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0).")
    parser.add_argument("--perms", type=int, default=200,
                        help="Tier 2b: label-permutation shuffles for the null (default 200).")
    parser.add_argument("--n-boot", dest="n_boot", type=int, default=2000,
                        help="Tier 2b: bootstrap resamples for the peak-AUC CIs (default 2000; 0 = off).")
    args = parser.parse_args()

    common = dict(output_dir=args.output, plots_dir=args.plots_dir, folds=args.folds,
                  seed=args.seed)
    if args.tier in ("2", "all"):
        from src.probe import run as run_tier2
        run_tier2(**common)
    if args.tier in ("2b", "all"):
        from src.probe_controls import run as run_tier2b
        run_tier2b(**common, perms=args.perms, n_boot=args.n_boot)
    if args.tier in ("2g", "all"):
        from src.probe_generalization import run as run_tier2g
        run_tier2g(**common)


if __name__ == "__main__":
    main()
