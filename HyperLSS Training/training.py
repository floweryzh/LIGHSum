from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import HyperLSSConfig, target_summary_sentences
from data import HyperLSSDataset, collate_hyperlss
from inference import rouge_f1, split_document_probabilities
from model import HyperLSS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def document_mean_loss(
    per_sentence_loss: torch.Tensor,
    document_ptr: torch.Tensor,
) -> torch.Tensor:
    """Average sentence loss within each document, then across documents.

    This follows Equation (33) of the paper, where each document contributes
    ``1 / n`` times the sum of its sentence losses rather than being weighted
    by its number of sentences in the minibatch.
    """
    pointers = document_ptr.detach().cpu().tolist()
    losses = [
        per_sentence_loss[start:end].mean()
        for start, end in zip(pointers[:-1], pointers[1:])
        if end > start
    ]
    if not losses:
        return per_sentence_loss.new_zeros(())
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    summary_length: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_documents = 0
    rouge_totals = np.zeros(3, dtype=np.float64)
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        batch_loss = document_mean_loss(
            criterion(logits, batch.labels), batch.document_ptr
        )
        probabilities = torch.sigmoid(logits)
        probability_lists = split_document_probabilities(
            probabilities, batch.document_ptr
        )
        scores = rouge_f1(
            probability_lists,
            batch.sentences,
            batch.references,
            summary_length,
        )
        document_count = len(batch.sentences)
        total_loss += float(batch_loss.item()) * document_count
        rouge_totals += np.asarray(scores) * document_count
        total_documents += document_count

    divisor = max(total_documents, 1)
    r1, r2, rl = (rouge_totals / divisor).tolist()
    return {
        "loss": total_loss / max(total_documents, 1),
        "rouge1": r1,
        "rouge2": r2,
        "rougeL": rl,
        "mean_rouge": (r1 + r2 + rl) / 3.0,
    }


def _state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()


def train_seed(
    train_dataset: HyperLSSDataset,
    validation_dataset: HyperLSSDataset,
    test_dataset: HyperLSSDataset,
    dataset_name: str,
    output_dir: str | Path,
    seed: int,
    *,
    num_workers: int = 4,
    config: HyperLSSConfig | None = None,
) -> dict[str, Any] | None:
    config = config or HyperLSSConfig()
    config.validate()
    rank, world_size, local_rank, device = distributed_context()
    if config.global_batch_size % world_size:
        raise ValueError("global_batch_size must be divisible by WORLD_SIZE")
    local_batch_size = config.global_batch_size // world_size
    set_seed(seed + rank)

    sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_hyperlss,
    )
    validation_loader = None
    test_loader = None
    if rank == 0:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.global_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_hyperlss,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.global_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_hyperlss,
        )

    base_model = HyperLSS(config).to(device)
    if world_size > 1:
        model: nn.Module = DistributedDataParallel(
            base_model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            # A filtered document batch can legitimately contain no edges of
            # one relation type, leaving that relation's parameters unused.
            find_unused_parameters=True,
        )
    else:
        model = base_model

    positive_weight = train_dataset.positive_class_weight()
    train_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device), reduction="none"
    )
    validation_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device), reduction="none"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    seed_dir = Path(output_dir) / f"seed-{seed}"
    checkpoint_path = seed_dir / "best.pt"
    if rank == 0:
        seed_dir.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    best_validation_score = -float("inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    summary_length = target_summary_sentences(dataset_name)

    for epoch in range(1, config.max_epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        running_loss = torch.zeros(2, device=device)
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = document_mean_loss(
                train_criterion(logits, batch.labels), batch.document_ptr
            )
            loss.backward()
            optimizer.step()
            running_loss[0] += loss.detach() * len(batch.sentences)
            running_loss[1] += len(batch.sentences)
        if world_size > 1:
            dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)

        stop = False
        if rank == 0:
            assert validation_loader is not None
            metrics = evaluate(
                base_model,
                validation_loader,
                validation_criterion,
                device,
                summary_length,
            )
            metrics.update(
                epoch=epoch,
                train_loss=float((running_loss[0] / running_loss[1].clamp_min(1)).item()),
            )
            history.append(metrics)
            if metrics["loss"] < best_validation_loss:
                best_validation_loss = metrics["loss"]
                stale_epochs = 0
            else:
                stale_epochs += 1
            # The paper selects the test checkpoint by mean validation ROUGE.
            if metrics["mean_rouge"] > best_validation_score:
                best_validation_score = metrics["mean_rouge"]
                torch.save(
                    {
                        "model": _state_dict(model),
                        "config": config.to_dict(),
                        "seed": seed,
                        "epoch": epoch,
                        "positive_class_weight": positive_weight,
                        "validation": metrics,
                    },
                    checkpoint_path,
                )
            stop = stale_epochs >= config.early_stopping_patience

        if world_size > 1:
            stop_tensor = torch.tensor(int(stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            stop = bool(stop_tensor.item())
            dist.barrier()
        if stop:
            break

    result = None
    if rank == 0:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        base_model.load_state_dict(checkpoint["model"])
        assert test_loader is not None
        test_metrics = evaluate(
            base_model, test_loader, validation_criterion, device, summary_length
        )
        result = {
            "seed": seed,
            "best_epoch": checkpoint["epoch"],
            "validation": checkpoint["validation"],
            "test": test_metrics,
            "history": history,
        }
        (seed_dir / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    if world_size > 1:
        dist.barrier()
    return result
