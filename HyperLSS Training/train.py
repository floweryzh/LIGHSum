from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch.distributed as dist

from data import HyperLSSDataset
from training import distributed_context, train_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the paper-aligned HyperLSS model")
    parser.add_argument("--dataset", required=True, choices=("pubmed", "arxiv"))
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    # The paper reports three runs but does not publish their numeric seed IDs.
    parser.add_argument("--seeds", nargs=3, type=int, default=(13, 42, 2024))
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dataset = HyperLSSDataset(args.train_manifest)
    validation_dataset = HyperLSSDataset(args.validation_manifest)
    test_dataset = HyperLSSDataset(args.test_manifest)
    rank, _, _, _ = distributed_context()
    results = []
    for seed in args.seeds:
        result = train_seed(
            train_dataset,
            validation_dataset,
            test_dataset,
            args.dataset,
            args.output_dir,
            seed,
            num_workers=args.num_workers,
        )
        if rank == 0 and result is not None:
            results.append(result)
    if rank == 0:
        means = {
            metric: sum(run["test"][metric] for run in results) / len(results)
            for metric in ("rouge1", "rouge2", "rougeL")
        }
        output = {"runs": results, "three_seed_mean": means}
        output_path = Path(args.output_dir) / "results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps(output["three_seed_mean"], indent=2))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

