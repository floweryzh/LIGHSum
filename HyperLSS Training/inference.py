from __future__ import annotations

import string
from collections.abc import Sequence

import torch


def _trigrams(text: str) -> set[tuple[str, str, str]]:
    table = str.maketrans("", "", string.punctuation)
    words = text.lower().translate(table).split()
    return set(zip(words, words[1:], words[2:]))


def select_summary_indices(
    probabilities: Sequence[float],
    sentences: Sequence[str],
    length_limit: int,
) -> list[int]:
    """Rank by salience, apply trigram blocking, then restore source order."""
    ranked = sorted(range(len(probabilities)), key=lambda i: (-float(probabilities[i]), i))
    selected: list[int] = []
    selected_trigrams: set[tuple[str, str, str]] = set()
    for index in ranked:
        candidate = _trigrams(sentences[index])
        if candidate & selected_trigrams:
            continue
        selected.append(index)
        selected_trigrams.update(candidate)
        if len(selected) == length_limit:
            break
    return sorted(selected)


def split_document_probabilities(
    probabilities: torch.Tensor, document_ptr: torch.Tensor
) -> list[list[float]]:
    values = probabilities.detach().cpu().tolist()
    ptr = document_ptr.detach().cpu().tolist()
    return [values[ptr[i] : ptr[i + 1]] for i in range(len(ptr) - 1)]


def rouge_f1(
    probability_lists: list[list[float]],
    sentence_lists: list[list[str]],
    reference_lists: list[list[str]],
    length_limit: int,
) -> tuple[float, float, float]:
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError("rouge-score is required for evaluation") from exc
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    totals = [0.0, 0.0, 0.0]
    for probabilities, sentences, references in zip(
        probability_lists, sentence_lists, reference_lists
    ):
        indices = select_summary_indices(probabilities, sentences, length_limit)
        candidate = " ".join(sentences[index] for index in indices)
        reference = " ".join(references)
        scores = scorer.score(reference, candidate)
        for i, name in enumerate(("rouge1", "rouge2", "rougeL")):
            totals[i] += scores[name].fmeasure
    count = max(len(probability_lists), 1)
    return tuple(total / count for total in totals)  # type: ignore[return-value]

