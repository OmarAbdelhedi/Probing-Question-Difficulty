"""FATHOM entry point — THIN launcher.

This module contains NO business logic. It only:
  1. parses CLI arguments,
  2. loads the YAML config,
  3. delegates to ``src.pipeline.run``.

All real source lives in the ``src/`` package and is imported with
absolute imports rooted at ``src`` (e.g. ``from src.pipeline import run``),
so that ``python main.py ...`` works directly without ``python -m``.
"""

import argparse

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "FATHOM: build recursive decomposition trees for labeled QA "
            "questions and extract per-layer LLM hidden-state activations."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="REQUIRED. Directory of labeled questions (*.jsonl files).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="REQUIRED. Output directory (created, incl. an activations/ subdir).",
    )
    parser.add_argument(
        "--config",
        default="./config.yaml",
        help="Path to the YAML experiment config (default: ./config.yaml).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N questions (default: all).",
    )
    parser.add_argument(
        "--max-depth",
        dest="max_depth",
        type=int,
        default=None,
        help="Override config max_depth (recursion hard cap).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Override config device.",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        default=None,
        help="Override config model_name (HF causal LM id used for BOTH "
             "decomposition and activation extraction).",
    )
    args = parser.parse_args()

    # Load experiment params (NO paths live in the config; paths come from CLI).
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Import here so argument parsing / config errors surface before the heavy
    # torch/transformers import chain. Absolute import rooted at src.
    from src.pipeline import run

    run(
        input_dir=args.input,
        output_dir=args.output,
        config=cfg,
        limit=args.limit,
        max_depth=args.max_depth,
        device=args.device,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
