"""Unified CLI for FATHOM dataset and transfer reports.

This replaces the scattered dataset/fused/pairwise plotting scripts with one
entrypoint:

    python transfer_suite.py dataset
    python transfer_suite.py pairwise
    python transfer_suite.py fused
"""

from __future__ import annotations

import argparse
import os

from src.probe_core import REPO_ROOT
from src.transfer_reports import (
    build_dataset_report,
    build_fused_reports,
    build_pairwise_reports,
)


def _shared_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--outputs-root",
        default=os.path.join(REPO_ROOT, "outputs"),
        help="Root containing run_* directories (default: <repo>/outputs).",
    )
    parent.add_argument(
        "--plots-root",
        default=os.path.join(REPO_ROOT, "figures"),
        help="Root where report figures are written (default: <repo>/figures).",
    )
    parent.add_argument("--folds", type=int, default=5, help="CV folds for within-dataset ceilings.")
    parent.add_argument("--seed", type=int, default=0, help="RNG seed for CV folds.")
    parent.add_argument("--top-k", type=int, default=10, help="Top dimensions to show in diagnostics.")
    parent.add_argument(
        "--report-level",
        choices=["summary", "diagnostics"],
        default="diagnostics",
        help="summary omits feature bars; diagnostics includes them (default).",
    )
    return parent


def main() -> None:
    shared = _shared_parent()
    parser = argparse.ArgumentParser(
        description="Build consolidated FATHOM dataset, pairwise transfer, and fused transfer reports."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dataset = sub.add_parser(
        "dataset",
        parents=[shared],
        help="Build figures/dataset_report.png and figures/dataset_report.json.",
    )
    dataset.add_argument(
        "--run-outputs",
        nargs="*",
        default=None,
        help="Exactly three run directories. Default: run_hotpot, run_2wiki, run_musique.",
    )

    pairwise = sub.add_parser(
        "pairwise",
        parents=[shared],
        help="Build one tier2t_transfer_report per pairwise transfer folder.",
    )
    pairwise.add_argument(
        "--transfer-dirs",
        nargs="*",
        default=None,
        help="Pairwise transfer folders. Default: figures/transfer_* with tier2t_summary.json.",
    )

    fused = sub.add_parser(
        "fused",
        parents=[shared],
        help="Run all 2-train/1-test fused reports.",
    )
    fused.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run directories to sweep. Default: run_2wiki, run_hotpot, run_musique.",
    )
    fused.add_argument("--dry-run", action="store_true", help="Plan fused jobs without computing reports.")

    args = parser.parse_args()

    if args.command == "dataset":
        build_dataset_report(
            run_outputs=args.run_outputs,
            outputs_root=args.outputs_root,
            plots_root=args.plots_root,
            folds=args.folds,
            seed=args.seed,
            top_k=args.top_k,
            report_level=args.report_level,
        )
    elif args.command == "pairwise":
        build_pairwise_reports(
            transfer_dirs=args.transfer_dirs,
            outputs_root=args.outputs_root,
            plots_root=args.plots_root,
            folds=args.folds,
            seed=args.seed,
            top_k=args.top_k,
            report_level=args.report_level,
        )
    elif args.command == "fused":
        build_fused_reports(
            runs=args.runs,
            outputs_root=args.outputs_root,
            plots_root=args.plots_root,
            folds=args.folds,
            seed=args.seed,
            top_k=args.top_k,
            report_level=args.report_level,
            dry_run=args.dry_run,
        )
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
