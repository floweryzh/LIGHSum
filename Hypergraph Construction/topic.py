# -*- coding: utf-8 -*-
"""Topic-hyperedge construction for HyperLSS.

LDA is fitted only on the training split of a dataset. The fitted dictionary
and model are then reused unchanged for training, validation, and test data.
Each sentence is assigned to the topic with the highest posterior probability,
and sentences assigned to the same topic form one topic hyperedge. Hyperedges
outside the configured membership interval are removed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TOPIC_COUNT = 50
DEFAULT_MIN_HYPEREDGE_SIZE = 2
DEFAULT_MAX_HYPEREDGE_SIZE = 25


@dataclass(frozen=True)
class TopicHyperedge:
    """One latent topic and its sentence-node members."""

    topic_id: int
    sentence_nodes: list[int]


def preprocess_sentence(sentence: str) -> list[str]:
    """Tokenize one sentence for LDA and remove English stop words."""
    try:
        from gensim.parsing.preprocessing import STOPWORDS
        from gensim.utils import simple_preprocess
    except ImportError as exc:
        raise RuntimeError("gensim is required for LDA preprocessing") from exc

    text = str(sentence).strip()
    if not text:
        return []
    return [
        token
        for token in simple_preprocess(text, deacc=True, min_len=2)
        if token not in STOPWORDS
    ]


def fit_training_lda(
    training_sentences: Iterable[str],
    *,
    topic_count: int = DEFAULT_TOPIC_COUNT,
    passes: int = 1,
    workers: int = 8,
    random_state: int = 13,
):
    """Fit the LDA dictionary and model using training sentences only."""
    try:
        from gensim.corpora import Dictionary
        from gensim.models import LdaMulticore
    except ImportError as exc:
        raise RuntimeError("gensim is required to fit LDA") from exc

    if topic_count <= 0:
        raise ValueError("topic_count must be positive")
    if passes <= 0 or workers <= 0:
        raise ValueError("passes and workers must be positive")

    tokenized_sentences = [
        tokens
        for sentence in training_sentences
        if (tokens := preprocess_sentence(sentence))
    ]
    if not tokenized_sentences:
        raise ValueError("training_sentences contains no usable LDA tokens")

    dictionary = Dictionary(tokenized_sentences)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_sentences]
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=topic_count,
        passes=passes,
        workers=workers,
        random_state=random_state,
        minimum_probability=0.0,
    )
    return dictionary, lda_model


def save_training_lda(
    dictionary: Any,
    lda_model: Any,
    output_dir: str | Path,
) -> None:
    """Save the training-fitted dictionary and LDA model for split reuse."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary.save(str(output_dir / "dictionary.gensim"))
    lda_model.save(str(output_dir / "lda.gensim"))


def load_training_lda(model_dir: str | Path):
    """Load the same training-fitted artifacts for any dataset split."""
    try:
        from gensim.corpora import Dictionary
        from gensim.models import LdaMulticore
    except ImportError as exc:
        raise RuntimeError("gensim is required to load LDA") from exc

    model_dir = Path(model_dir)
    dictionary = Dictionary.load(str(model_dir / "dictionary.gensim"))
    lda_model = LdaMulticore.load(str(model_dir / "lda.gensim"))
    return dictionary, lda_model


def assign_sentence_topics(
    sentences: Sequence[str],
    dictionary: Any,
    lda_model: Any,
) -> np.ndarray:
    """Assign each sentence to its maximum-posterior LDA topic."""
    topic_ids: list[int] = []
    expected_topic_count = int(lda_model.num_topics)

    for sentence in sentences:
        tokens = preprocess_sentence(sentence)
        bow = dictionary.doc2bow(tokens)
        posterior = lda_model.get_document_topics(bow, minimum_probability=0.0)
        if len(posterior) != expected_topic_count:
            probabilities = {int(topic): float(probability) for topic, probability in posterior}
            posterior = [
                (topic, probabilities.get(topic, 0.0))
                for topic in range(expected_topic_count)
            ]
        topic_id, _ = max(posterior, key=lambda item: (item[1], -item[0]))
        topic_ids.append(int(topic_id))

    return np.asarray(topic_ids, dtype=np.int16)


def topic_hyperedges_from_assignments(
    topic_ids: Sequence[int] | np.ndarray,
    *,
    min_hyperedge_size: int = DEFAULT_MIN_HYPEREDGE_SIZE,
    max_hyperedge_size: int = DEFAULT_MAX_HYPEREDGE_SIZE,
) -> list[TopicHyperedge]:
    """Group same-topic sentence nodes and apply the paper's size filter."""
    if min_hyperedge_size < 1 or min_hyperedge_size > max_hyperedge_size:
        raise ValueError("invalid hyperedge-size interval")

    members_by_topic: dict[int, list[int]] = defaultdict(list)
    for sentence_node, topic_id in enumerate(topic_ids):
        members_by_topic[int(topic_id)].append(sentence_node)

    hyperedges: list[TopicHyperedge] = []
    for topic_id in sorted(members_by_topic):
        members = members_by_topic[topic_id]
        if min_hyperedge_size <= len(members) <= max_hyperedge_size:
            hyperedges.append(
                TopicHyperedge(topic_id=topic_id, sentence_nodes=members)
            )
    return hyperedges


def build_topic_hyperedges(
    sentences: Sequence[str],
    dictionary: Any,
    lda_model: Any,
    *,
    min_hyperedge_size: int = DEFAULT_MIN_HYPEREDGE_SIZE,
    max_hyperedge_size: int = DEFAULT_MAX_HYPEREDGE_SIZE,
) -> tuple[np.ndarray, list[TopicHyperedge]]:
    """Assign topics and construct one document's retained topic hyperedges."""
    topic_ids = assign_sentence_topics(sentences, dictionary, lda_model)
    hyperedges = topic_hyperedges_from_assignments(
        topic_ids,
        min_hyperedge_size=min_hyperedge_size,
        max_hyperedge_size=max_hyperedge_size,
    )
    return topic_ids, hyperedges


def topic_incidence_matrix(
    sentence_count: int,
    hyperedges: Sequence[TopicHyperedge],
) -> np.ndarray:
    """Create the binary sentence-by-topic-hyperedge incidence matrix."""
    if sentence_count < 0:
        raise ValueError("sentence_count must be non-negative")
    incidence = np.zeros((sentence_count, len(hyperedges)), dtype=np.int8)
    for hyperedge_index, hyperedge in enumerate(hyperedges):
        for sentence_node in hyperedge.sentence_nodes:
            if not 0 <= sentence_node < sentence_count:
                raise ValueError("hyperedge contains an invalid sentence node")
            incidence[sentence_node, hyperedge_index] = 1
    return incidence

