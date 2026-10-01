# -*- coding: utf-8 -*-
"""Keyword-hyperedge construction for HyperLSS.

KeyBERT extracts the top K_key keywords or keyphrases from a document. Each
keyword/keyphrase and every sentence are represented by the same pretrained
all-MiniLM-L6-v2 encoder. For each keyword, the K_sent sentences with the
highest cosine similarity are connected by one keyword hyperedge. Hyperedges
outside the configured membership interval are removed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_KEYWORD_COUNT = 15
DEFAULT_SENTENCES_PER_KEYWORD = 15
DEFAULT_MIN_HYPEREDGE_SIZE = 2
DEFAULT_MAX_HYPEREDGE_SIZE = 25


@dataclass(frozen=True)
class KeywordHyperedge:
    """One keyword/keyphrase and its selected sentence members."""

    keyword: str
    sentence_nodes: list[int]
    similarities: list[float]


def load_keyword_models(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
):
    """Load one shared sentence encoder and the KeyBERT wrapper."""
    try:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "keybert and sentence-transformers are required for keyword hyperedges"
        ) from exc

    sentence_model = SentenceTransformer(model_name, device=device)
    keyword_model = KeyBERT(model=sentence_model)
    return sentence_model, keyword_model


def _encode(model, texts: Sequence[str], batch_size: int) -> np.ndarray:
    """Encode text without assuming CUDA is available."""
    if not texts:
        dimension = int(model.get_sentence_embedding_dimension())
        return np.empty((0, dimension), dtype=np.float32)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def _cosine_cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return all cosine similarities between two embedding matrices."""
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("embedding matrices must be two-dimensional")
    if left.shape[1] != right.shape[1]:
        raise ValueError("keyword and sentence embedding dimensions must match")

    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left_unit = np.divide(left, left_norm, out=np.zeros_like(left), where=left_norm > 0)
    right_unit = np.divide(
        right, right_norm, out=np.zeros_like(right), where=right_norm > 0
    )
    return np.clip(left_unit @ right_unit.T, -1.0, 1.0)


def extract_document_keywords(
    sentences: Sequence[str],
    keyword_model,
    *,
    keyword_count: int = DEFAULT_KEYWORD_COUNT,
    keyphrase_ngram_range: tuple[int, int] = (1, 3),
) -> list[str]:
    """Extract the document's top keywords or keyphrases with KeyBERT."""
    if keyword_count <= 0:
        raise ValueError("keyword_count must be positive")
    sentence_list = [str(sentence).strip() for sentence in sentences]
    if not sentence_list or any(not sentence for sentence in sentence_list):
        raise ValueError("sentences must contain non-empty text")

    document_text = " ".join(sentence_list)
    extracted = keyword_model.extract_keywords(
        document_text,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words="english",
        top_n=keyword_count,
        use_mmr=False,
    )
    return [str(keyword) for keyword, _ in extracted]


def keyword_hyperedges_from_embeddings(
    keywords: Sequence[str],
    keyword_embeddings: np.ndarray,
    sentence_embeddings: np.ndarray,
    *,
    sentences_per_keyword: int = DEFAULT_SENTENCES_PER_KEYWORD,
    min_hyperedge_size: int = DEFAULT_MIN_HYPEREDGE_SIZE,
    max_hyperedge_size: int = DEFAULT_MAX_HYPEREDGE_SIZE,
) -> list[KeywordHyperedge]:
    """Select top-similarity sentence members for every keyword hyperedge."""
    if len(keywords) != len(keyword_embeddings):
        raise ValueError("keywords and keyword_embeddings must have equal lengths")
    if sentences_per_keyword <= 0:
        raise ValueError("sentences_per_keyword must be positive")
    if min_hyperedge_size < 1 or min_hyperedge_size > max_hyperedge_size:
        raise ValueError("invalid hyperedge-size interval")
    if len(sentence_embeddings) == 0:
        return []

    similarities = _cosine_cross(keyword_embeddings, sentence_embeddings)
    selected_count = min(sentences_per_keyword, len(sentence_embeddings))
    hyperedges: list[KeywordHyperedge] = []

    for keyword, scores in zip(keywords, similarities):
        # Descending score, with sentence index as a deterministic tie-breaker.
        ranked = np.lexsort((np.arange(len(scores)), -scores))[:selected_count]
        members = ranked.astype(int).tolist()
        if not min_hyperedge_size <= len(members) <= max_hyperedge_size:
            continue
        hyperedges.append(
            KeywordHyperedge(
                keyword=str(keyword),
                sentence_nodes=members,
                similarities=[float(scores[index]) for index in ranked],
            )
        )
    return hyperedges


def build_keyword_hyperedges(
    sentences: Sequence[str],
    sentence_model=None,
    keyword_model=None,
    *,
    sentence_embeddings: np.ndarray | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    batch_size: int = 64,
    keyword_count: int = DEFAULT_KEYWORD_COUNT,
    sentences_per_keyword: int = DEFAULT_SENTENCES_PER_KEYWORD,
    min_hyperedge_size: int = DEFAULT_MIN_HYPEREDGE_SIZE,
    max_hyperedge_size: int = DEFAULT_MAX_HYPEREDGE_SIZE,
    keyphrase_ngram_range: tuple[int, int] = (1, 3),
) -> list[KeywordHyperedge]:
    """Extract keywords and construct one document's keyword hyperedges."""
    if sentence_model is None:
        sentence_model, loaded_keyword_model = load_keyword_models(
            model_name=model_name,
            device=device,
        )
        if keyword_model is None:
            keyword_model = loaded_keyword_model
    elif keyword_model is None:
        try:
            from keybert import KeyBERT
        except ImportError as exc:
            raise RuntimeError("keybert is required for keyword hyperedges") from exc
        keyword_model = KeyBERT(model=sentence_model)

    sentence_list = [str(sentence).strip() for sentence in sentences]
    keywords = extract_document_keywords(
        sentence_list,
        keyword_model,
        keyword_count=keyword_count,
        keyphrase_ngram_range=keyphrase_ngram_range,
    )
    if not keywords:
        return []

    if sentence_embeddings is None:
        sentence_embeddings = _encode(sentence_model, sentence_list, batch_size)
    else:
        sentence_embeddings = np.asarray(sentence_embeddings, dtype=np.float32)
        if len(sentence_embeddings) != len(sentence_list):
            raise ValueError(
                "sentence_embeddings must contain one vector per sentence"
            )
    keyword_embeddings = _encode(sentence_model, keywords, batch_size)

    return keyword_hyperedges_from_embeddings(
        keywords,
        keyword_embeddings,
        sentence_embeddings,
        sentences_per_keyword=sentences_per_keyword,
        min_hyperedge_size=min_hyperedge_size,
        max_hyperedge_size=max_hyperedge_size,
    )


def keyword_incidence_matrix(
    sentence_count: int,
    hyperedges: Sequence[KeywordHyperedge],
) -> np.ndarray:
    """Create the binary sentence-by-keyword-hyperedge incidence matrix."""
    if sentence_count < 0:
        raise ValueError("sentence_count must be non-negative")
    incidence = np.zeros((sentence_count, len(hyperedges)), dtype=np.int8)
    for hyperedge_index, hyperedge in enumerate(hyperedges):
        for sentence_node in hyperedge.sentence_nodes:
            if not 0 <= sentence_node < sentence_count:
                raise ValueError("hyperedge contains an invalid sentence node")
            incidence[sentence_node, hyperedge_index] = 1
    return incidence
