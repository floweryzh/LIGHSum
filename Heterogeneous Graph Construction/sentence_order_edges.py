# -*- coding: utf-8 -*-
"""Sentence-order relation construction for HyperLSS.

For each pair of consecutive sentences, HyperLSS creates one directed edge
from the preceding sentence to the following sentence. Every sentence-order
edge has a fixed weight of 1.0. Reverse edges and self-loops are not added.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


SENTENCE_ORDER_WEIGHT = 1.0


@dataclass(frozen=True)
class SentenceOrderRelations:
    """Weighted directed sentence-order edges for one document."""

    edges: list[tuple[tuple[int, int], float]]


def sentence_order_edges(sentence_count: int) -> SentenceOrderRelations:
    """Create ``s_i -> s_(i+1)`` edges with fixed weight 1.0.

    Sentence nodes are numbered from ``0`` through ``sentence_count - 1``.
    A document with zero or one sentence has no sentence-order edges.
    """
    if isinstance(sentence_count, bool) or not isinstance(sentence_count, int):
        raise TypeError("sentence_count must be an integer")
    if sentence_count < 0:
        raise ValueError("sentence_count must be non-negative")

    edges = [
        ((source, source + 1), SENTENCE_ORDER_WEIGHT)
        for source in range(sentence_count - 1)
    ]
    return SentenceOrderRelations(edges=edges)


def build_sentence_order_edges(
    sentences: Sequence[str],
) -> SentenceOrderRelations:
    """Create sentence-order edges directly from a document's sentence list."""
    sentence_list = [str(sentence).strip() for sentence in sentences]
    if any(not sentence for sentence in sentence_list):
        raise ValueError("sentences must not contain empty text")
    return sentence_order_edges(len(sentence_list))

