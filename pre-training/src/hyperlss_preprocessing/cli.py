from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cleaning import CorpusCleaner
from .config import PreprocessConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hyperlss-preprocess",
        description="HyperLSS text cleaning, filtering, and oracle-label preprocessing",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean = subparsers.add_parser("clean", help="normalize and filter one raw JSONL split")
    clean.add_argument("--dataset", required=True, choices=("pubmed", "arxiv"))
    clean.add_argument("--input", required=True)
    clean.add_argument("--output", required=True)

    run = subparsers.add_parser("run", help="preprocess train/validation/test splits")
    run.add_argument("--dataset", required=True, choices=("pubmed", "arxiv"))
    run.add_argument("--train", required=True)
    run.add_argument("--validation", required=True)
    run.add_argument("--test", required=True)
    run.add_argument("--output-dir", required=True)
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = PreprocessConfig.for_dataset(args.dataset)
    if args.command == "clean":
        _print(CorpusCleaner(config).clean_file(args.input, args.output))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaner = CorpusCleaner(config)
    raw_paths = {"train": args.train, "validation": args.validation, "test": args.test}
    stats = {}
    for split, raw_path in raw_paths.items():
        stats[split] = cleaner.clean_file(raw_path, output_dir / f"{split}.jsonl")
    _print(stats)


if __name__ == "__main__":
    main()
