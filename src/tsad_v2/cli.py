from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsad-v2", description="TSAD v2 command-line pipeline")
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="YAML config; repeat to overlay configs (default: configs/base.yaml)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("generate-synthetic", help="generate train/val synthetic data")
    subparsers.add_parser(
        "prepare-ucr", help="create one full test-region inference sample per UCR series"
    )

    validate = subparsers.add_parser("validate-manifest", help="validate a training manifest")
    validate.add_argument("manifest", type=Path)

    infer = subparsers.add_parser("infer", help="run local text or vision model inference")
    infer.add_argument("manifest", type=Path)
    infer.add_argument("output", type=Path)
    infer.add_argument("--model")
    infer.add_argument("--adapter")
    infer.add_argument("--modality", required=True, choices=("text", "vision"))
    infer.add_argument("--limit", type=int)

    isolation = subparsers.add_parser("infer-isolation-forest", help="run the numeric baseline")
    isolation.add_argument("manifest", type=Path)
    isolation.add_argument("output", type=Path)
    isolation.add_argument("--limit", type=int)

    evaluate = subparsers.add_parser("evaluate", help="evaluate predictions with the unified protocol")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("output_dir", type=Path)

    compare = subparsers.add_parser("compare", help="compare C0/T0/V0/T1/V1 summary files")
    compare.add_argument("reports", nargs="+", type=Path)
    compare.add_argument("--output", required=True, type=Path)

    sft = subparsers.add_parser("train-sft", help="train a text or vision QLoRA SFT adapter")
    sft.add_argument("--modality", required=True, choices=("text", "vision"))
    sft.add_argument("--resume")
    sft.add_argument("--limit", type=int)
    return parser


def _config(paths: List[str]) -> Dict[str, Any]:
    return load_config(paths or ["configs/base.yaml"])


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    config = _config(args.config)
    if args.command == "generate-synthetic":
        from .data.synthetic import generate_dataset

        paths = generate_dataset(
            config["synthetic"],
            config.get("render", {}),
            int(config["project"].get("seed", 3407)),
            config["ucr"],
        )
        print("\n".join(str(path) for path in paths))
    elif args.command == "prepare-ucr":
        from .data.ucr import prepare_ucr

        print(prepare_ucr(config["ucr"], config.get("render", {})))
    elif args.command == "validate-manifest":
        from .data.common import load_training_manifest

        records = load_training_manifest(args.manifest)
        print(json.dumps({"manifest": str(args.manifest), "records": len(records)}, indent=2))
    elif args.command == "infer":
        from .inference import run_inference

        run_inference(
            config, args.manifest, args.output, args.modality, args.model, args.adapter, args.limit
        )
    elif args.command == "infer-isolation-forest":
        from .baselines.isolation_forest import run_isolation_forest

        run_isolation_forest(config, args.manifest, args.output, args.limit)
    elif args.command == "evaluate":
        from .evaluation import evaluate_predictions

        print(json.dumps(evaluate_predictions(args.manifest, args.predictions, args.output_dir), indent=2))
    elif args.command == "compare":
        from .evaluation import compare_reports

        compare_reports(args.reports, args.output)
        print(args.output)
    elif args.command == "train-sft":
        from .training.sft import train_sft

        print(train_sft(config, args.modality, args.resume, args.limit))
    else:
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
