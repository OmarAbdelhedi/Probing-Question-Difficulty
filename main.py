"""Extract per-layer hidden states for a directory of labeled questions.

    python main.py --input data/questions_musique \\
        --output outputs/qwen32b/run_musique --model Qwen/Qwen2.5-32B-Instruct

One forward pass per question, no generation. Re-running with the same
``--output`` resumes: questions whose ``activations/<id>.npz`` already exists
are skipped. See ``src/pipeline.py`` for the output format.
"""

import argparse

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-layer hidden states of labeled questions from a causal LM."
    )
    parser.add_argument("--input", required=True,
                        help="Directory of labeled questions (*.jsonl with id/question/label).")
    parser.add_argument("--output", required=True,
                        help="Run directory to write activations/, metadata.jsonl and run_config.json to.")
    parser.add_argument("--config", default="config.yaml",
                        help="YAML extraction settings (default: config.yaml).")
    parser.add_argument("--model", dest="model_name", default=None,
                        help="HuggingFace model id; overrides model_name in the config.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                        help="Overrides device in the config.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N questions.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # Imported here so argument errors surface before the torch import.
    from src.pipeline import run

    run(
        input_dir=args.input,
        output_dir=args.output,
        config=config,
        limit=args.limit,
        device=args.device,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
