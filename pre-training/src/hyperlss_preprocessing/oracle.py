from __future__ import annotations


def greedy_oracle_labels(
    sentences: list[str], reference_sentences: list[str], length_limit: int
) -> tuple[list[int], list[int], float]:
    """Equation (31)-(32): greedily maximize R-1 + R-2 + R-L F1."""
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("rouge-score is required to construct oracle labels") from exc

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    reference = " ".join(reference_sentences)

    def objective(indices: list[int]) -> float:
        candidate = " ".join(sentences[index] for index in sorted(indices))
        scores = scorer.score(reference, candidate)
        return sum(scores[name].fmeasure for name in ("rouge1", "rouge2", "rougeL"))

    selected: list[int] = []
    current_score = 0.0
    remaining = set(range(len(sentences)))
    while remaining and len(selected) < length_limit:
        best_index = -1
        best_score = current_score
        for index in sorted(remaining):
            score = objective(selected + [index])
            if score > best_score + 1e-12:
                best_index, best_score = index, score
        if best_index < 0:
            break
        selected.append(best_index)
        remaining.remove(best_index)
        current_score = best_score

    labels = [0] * len(sentences)
    for index in selected:
        labels[index] = 1
    return labels, sorted(selected), current_score

