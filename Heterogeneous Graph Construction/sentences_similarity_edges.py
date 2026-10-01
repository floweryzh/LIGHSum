# -*- coding: utf-8 -*-
"""Sentence-similarity relation construction for HyperLSS.

For every sentence pair ``i < j``, the cosine similarity of the original
all-MiniLM-L6-v2 representations is computed. If the score is at least
``tau_s`` (0.7 in the paper), two directed edges are created: ``i -> j``
and ``j -> i``. Both directions use the same cosine-similarity weight.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SENTENCE_SIMILARITY_THRESHOLD = 0.7


def load_sentence_encoder(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
):
    """Load the sentence encoder once and return it for reuse."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for sentence encoding"
        ) from exc

    return SentenceTransformer(model_name, device=device)


def Sentences_vector_embedding(
    sentences: Sequence[str],
    model=None,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode sentences with all-MiniLM-L6-v2.

    The returned array has shape ``[number_of_sentences, 384]`` for the
    default model. A caller-supplied model is reused instead of loading a
    model for every function call.
    """
    if model is None:
        model = load_sentence_encoder(model_name=model_name, device=device)

    sentence_list = [str(sentence).strip() for sentence in sentences]
    if any(not sentence for sentence in sentence_list):
        raise ValueError("sentences must not contain empty text")
    if not sentence_list:
        dimension = int(model.get_sentence_embedding_dimension())
        return np.empty((0, dimension), dtype=np.float32)

    embeddings = model.encode(
        sentence_list,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def Sentences_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Return the complete cosine-similarity matrix for sentence vectors."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = np.divide(
        embeddings,
        norms,
        out=np.zeros_like(embeddings),
        where=norms > 0,
    )
    similarities = normalized @ normalized.T
    return np.clip(similarities, -1.0, 1.0).astype(np.float32, copy=False)


def get_feature_similarity(
    similarity_matrix: np.ndarray,
    threshold: float = DEFAULT_SENTENCE_SIMILARITY_THRESHOLD,
) -> list[tuple[tuple[int, int], float]]:
    """Construct the paper's bidirectional sentence-similarity edges.

    Returns ``[((source, target), weight), ...]``. Self-loops are not added.
    Unlike the original window-based implementation, every pair ``i < j``
    is considered and the threshold comparison is inclusive.
    """
    similarities = np.asarray(similarity_matrix, dtype=np.float32)
    if similarities.ndim != 2 or similarities.shape[0] != similarities.shape[1]:
        raise ValueError("similarity_matrix must be square")
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between -1 and 1")

    edges: list[tuple[tuple[int, int], float]] = []
    sentence_count = similarities.shape[0]
    for source in range(sentence_count):
        for target in range(source + 1, sentence_count):
            score = float(similarities[source, target])
            if score >= threshold:
                edges.append(((source, target), score))
                edges.append(((target, source), score))
    return edges


def build_sentence_similarity_edges(
    sentences: Sequence[str],
    model=None,
    *,
    threshold: float = DEFAULT_SENTENCE_SIMILARITY_THRESHOLD,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    batch_size: int = 64,
) -> tuple[np.ndarray, list[tuple[tuple[int, int], float]]]:
    """Encode sentences and construct weighted similarity edges in one call."""
    embeddings = Sentences_vector_embedding(
        sentences,
        model=model,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    similarities = Sentences_similarity(embeddings)
    edges = get_feature_similarity(similarities, threshold=threshold)
    return embeddings, edges

