# -*- coding: utf-8 -*-
"""Evaluate a trained HyperLSS checkpoint on PubMed or arXiv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
TRAINING_DIR = ROOT / "HyperLSS Training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from config import HyperLSSConfig, target_summary_sentences  # noqa: E402
from data import HyperLSSDataset, collate_hyperlss  # noqa: E402
from inference import select_summary_indices, split_document_probabilities  # noqa: E402
from model import HyperLSS  # noqa: E402


def document_mean_loss(
    per_sentence_loss: torch.Tensor,
    document_ptr: torch.Tensor,
) -> torch.Tensor:
    """Match the paper's per-document normalization in Equation (33)."""
    pointers = document_ptr.detach().cpu().tolist()
    losses = [
        per_sentence_loss[start:end].mean()
        for start, end in zip(pointers[:-1], pointers[1:])
        if end > start
    ]
    if not losses:
        return per_sentence_loss.new_zeros(())
    return torch.stack(losses).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a HyperLSS checkpoint")
    parser.add_argument("--dataset", required=True, choices=("pubmed", "arxiv"))
    parser.add_argument("--manifest", required=True, help="test manifest.jsonl")
    parser.add_argument("--checkpoint", required=True, help="best.pt from training")
    parser.add_argument("--output", required=True, help="evaluation JSON output")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="for example: cuda, cuda:0, or cpu",
    )
    return parser.parse_args()


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[HyperLSS, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" not in checkpoint:
        raise ValueError("checkpoint does not contain a model state")
    config = HyperLSSConfig(**checkpoint.get("config", {}))
    config.validate()
    model = HyperLSS(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate_checkpoint(
    model: HyperLSS,
    checkpoint: dict,
    loader: DataLoader,
    dataset_name: str,
    device: torch.device,
) -> dict:
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError("rouge-score is required for evaluation") from exc

    positive_weight = float(checkpoint.get("positive_class_weight", 1.0))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device), reduction="none"
    )
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    length_limit = target_summary_sentences(dataset_name)
    total_loss = 0.0
    total_sentences = 0
    rouge_totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    documents: list[dict] = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        batch_loss = document_mean_loss(
            criterion(logits, batch.labels), batch.document_ptr
        )
        total_loss += float(batch_loss.item()) * len(batch.sentences)
        total_sentences += len(batch.sentences)
        probability_lists = split_document_probabilities(
            torch.sigmoid(logits), batch.document_ptr
        )
        for document_id, probabilities, sentences, references in zip(
            batch.document_ids,
            probability_lists,
            batch.sentences,
            batch.references,
        ):
            selected = select_summary_indices(probabilities, sentences, length_limit)
            summary_sentences = [sentences[index] for index in selected]
            summary = " ".join(summary_sentences)
            reference = " ".join(references)
            scores = scorer.score(reference, summary)
            document_scores = {
                name: float(scores[name].fmeasure)
                for name in ("rouge1", "rouge2", "rougeL")
            }
            for name, value in document_scores.items():
                rouge_totals[name] += value
            documents.append(
                {
                    "document_id": document_id,
                    "selected_sentence_indices": selected,
                    "summary_sentences": summary_sentences,
                    "reference_sentences": references,
                    "rouge": document_scores,
                }
            )

    document_count = max(len(documents), 1)
    aggregate = {
        name: total / document_count for name, total in rouge_totals.items()
    }
    aggregate["mean_rouge"] = sum(aggregate.values()) / 3.0
    aggregate["weighted_bce_loss"] = total_loss / max(total_sentences, 1)
    return {
        "dataset": dataset_name,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "seed": checkpoint.get("seed"),
        "summary_length": length_limit,
        "document_count": len(documents),
        "metrics": aggregate,
        "documents": documents,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = HyperLSSDataset(args.manifest)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_hyperlss,
    )
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    result = evaluate_checkpoint(model, checkpoint, loader, args.dataset, device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()

