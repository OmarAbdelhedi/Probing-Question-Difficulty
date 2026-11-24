"""Cross-dataset transfer of the difficulty axis (CPU only; Table II, Fig. 6).

    python transfer.py --runs outputs/qwen32b/run_2wiki outputs/qwen32b/run_hotpot \\
        outputs/qwen32b/run_musique --plots-root figures/qwen32b

Runs every combination that trains on two runs and tests on the third, writing
to ``<plots-root>/transfer_sweep/``. All runs must come from the same model,
pooling and layer selection. See ``src/transfer.py``.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the difficulty axis on two runs and test it on the held-out third."
    )
    parser.add_argument("--runs", nargs="+", required=True,
                        help="At least three run directories written by main.py.")
    parser.add_argument("--plots-root", required=True,
                        help="Parent directory of transfer_sweep/ (e.g. figures/qwen32b).")
    parser.add_argument("--folds", type=int, default=5,
                        help="CV folds for the within-dataset ceilings (default 5).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0).")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Feature dimensions listed in the diagnostics (default 10).")
    parser.add_argument("--report-level", choices=["summary", "diagnostics"], default="diagnostics",
                        help="'diagnostics' (default) adds per-dimension bar plots under Fig. 6's two panels.")
    args = parser.parse_args()

    from src.transfer import run

    run(runs=args.runs, plots_root=args.plots_root, folds=args.folds, seed=args.seed,
        top_k=args.top_k, report_level=args.report_level)


if __name__ == "__main__":
    main()
