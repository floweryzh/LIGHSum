# -*- coding: utf-8 -*-
"""Word-similarity relation construction for HyperLSS.

For every pair of distinct document word nodes, cosine similarity is computed
from the original 300-dimensional GloVe representations. If the score is at
least ``tau_w`` (0.6 in the paper), two directed edges are created. Both
directions use the same cosine-similarity weight.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np


GLOVE_DIMENSION = 300
DEFAULT_WORD_SIMILARITY_THRESHOLD = 0.6


@dataclass(frozen=True)
class WordSimilarityRelations:
    """Weighted directed word-similarity edges for one document."""

    edges: list[tuple[tuple[int, int], float]]
    missing_words: list[str]


def load_glove_vectors(
    glove_path: str | Path,
    vocabulary: Collection[str] | None = None,
    *,
    dimension: int = GLOVE_DIMENSION,
) -> dict[str, np.ndarray]:
    """Load GloVe text vectors, optionally restricted to required words.

    Passing the corpus vocabulary is recommended because it avoids keeping the
    complete GloVe table in memory. The same returned dictionary can be reused
    for every document in the corpus.
    """
    required = set(vocabulary) if vocabulary is not None else None
    vectors: dict[str, np.ndarray] = {}

    with Path(glove_path).open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.rstrip().split()
            if len(fields) != dimension + 1:
                continue
            word = fields[0]
            if required is not None and word not in required:
                continue
            try:
                vector = np.asarray(fields[1:], dtype=np.float32)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed GloVe vector at line {line_number}"
                ) from exc
            vectors[word] = vector
            if required is not None and len(vectors) == len(required):
                break
    return vectors


def cosine_similarity(source: np.ndarray, destination: np.ndarray) -> float:
    """Compute cosine similarity and safely handle zero vectors."""
    source = np.asarray(source, dtype=np.float32)
    destination = np.asarray(destination, dtype=np.float32)
    if source.ndim != 1 or destination.ndim != 1:
        raise ValueError("word vectors must be one-dimensional")
    if source.shape != destination.shape:
        raise ValueError("word vectors must have the same dimension")

    denominator = float(np.linalg.norm(source) * np.linalg.norm(destination))
    if denominator == 0.0:
        return 0.0
    score = float(np.dot(source, destination) / denominator)
    return float(np.clip(score, -1.0, 1.0))


def Words_similarity(
    source_word: str,
    destination_word: str,
    glove_vectors: Mapping[str, np.ndarray],
) -> float | None:
    """Return the GloVe cosine similarity, or None for an unknown word."""
    source_vector = glove_vectors.get(source_word)
    destination_vector = glove_vectors.get(destination_word)
    if source_vector is None or destination_vector is None:
        return None
    return cosine_similarity(source_vector, destination_vector)


def word_similarity_edges(
    words: Sequence[str],
    word_to_node: Mapping[str, int],
    glove_vectors: Mapping[str, np.ndarray],
    *,
    threshold: float = DEFAULT_WORD_SIMILARITY_THRESHOLD,
) -> WordSimilarityRelations:
    """Construct the paper's bidirectional word-similarity relations.

    `word_to_node` should be the document-local mapping produced by the
    word-sentence construction stage. This preserves its sentence-node offset
    and prevents word/sentence node-ID collisions.
    """
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between -1 and 1")
    if len(set(words)) != len(words):
        raise ValueError("words must contain distinct document word nodes")

    missing_mapping = [word for word in words if word not in word_to_node]
    if missing_mapping:
        raise ValueError(f"word_to_node is missing words: {missing_mapping[:5]}")

    missing_words = [word for word in words if word not in glove_vectors]
    known_words = [word for word in words if word in glove_vectors]
    invalid_dimensions = [
        word
        for word in known_words
        if np.asarray(glove_vectors[word]).shape != (GLOVE_DIMENSION,)
    ]
    if invalid_dimensions:
        raise ValueError(
            "HyperLSS requires 300-dimensional GloVe vectors; invalid words: "
            f"{invalid_dimensions[:5]}"
        )
    edges: list[tuple[tuple[int, int], float]] = []

    for source_word, destination_word in combinations(known_words, 2):
        score = Words_similarity(source_word, destination_word, glove_vectors)
        if score is not None and score >= threshold:
            source_node = int(word_to_node[source_word])
            destination_node = int(word_to_node[destination_word])
            edges.append(((source_node, destination_node), score))
            edges.append(((destination_node, source_node), score))

    return WordSimilarityRelations(edges=edges, missing_words=missing_words)


def build_word_similarity_edges(
    words: Sequence[str],
    sentence_count: int,
    glove_vectors: Mapping[str, np.ndarray],
    *,
    threshold: float = DEFAULT_WORD_SIMILARITY_THRESHOLD,
) -> WordSimilarityRelations:
    """Build edges when a word-node mapping has not already been created."""
    if sentence_count < 0:
        raise ValueError("sentence_count must be non-negative")
    word_to_node = {
        word: sentence_count + local_index for local_index, word in enumerate(words)
    }
    return word_similarity_edges(
        words,
        word_to_node,
        glove_vectors,
        threshold=threshold,
    )
