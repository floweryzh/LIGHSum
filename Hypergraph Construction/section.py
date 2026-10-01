# -*- coding: utf-8 -*-
"""Section-hyperedge construction for HyperLSS.

Section hyperedges are derived directly from the original document structure.
All sentences belonging to the same source section are connected by one
hyperedge. No named-entity recognition or semantic clustering is used.
Hyperedges outside the configured membership interval are removed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_MIN_HYPEREDGE_SIZE = 2
DEFAULT_MAX_HYPEREDGE_SIZE = 25


@dataclass(frozen=True)
class SectionHyperedge:
    """One original document section and its sentence-node members."""

    section_id: int
    section_name: str
    sentence_nodes: list[int]


def section_hyperedges_from_ids(
    section_ids: Sequence[int],
    section_names: Sequence[str] | Mapping[int, str] | None = None,
    *,
    min_hyperedge_size: int = DEFAULT_MIN_HYPEREDGE_SIZE,
    max_hyperedge_size: int = DEFAULT_MAX_HYPEREDGE_SIZE,
) -> list[SectionHyperedge]:
    """Group sentence nodes by their original section IDs."""
    if min_hyperedge_size < 1 or min_hyperedge_size > max_hyperedge_size:
        raise ValueError("invalid hyperedge-size interval")

    members_by_section: dict[int, list[int]] = defaultdict(list)
    for sentence_node, raw_section_id in enumerate(section_ids):
        if isinstance(raw_section_id, bool) or not isinstance(raw_section_id, (int, np.integer)):
            raise TypeError("section IDs must be integers")
        section_id = int(raw_section_id)
        if section_id < 0:
            raise ValueError("section IDs must be non-negative")
        members_by_section[section_id].append(sentence_node)

    def name_for(section_id: int) -> str:
        if section_names is None:
            return f"section-{section_id}"
        if isinstance(section_names, Mapping):
            return str(section_names.get(section_id, f"section-{section_id}"))
        if section_id >= len(section_names):
            raise ValueError(f"missing name for section ID {section_id}")
        return str(section_names[section_id])

    hyperedges: list[SectionHyperedge] = []
    for section_id in sorted(members_by_section):
        members = members_by_section[section_id]
        if min_hyperedge_size <= len(members) <= max_hyperedge_size:
            hyperedges.append(
                SectionHyperedge(
                    section_id=section_id,
                    section_name=name_for(section_id),
                    sentence_nodes=members,
                )
            )
    return hyperedges


def section_ids_from_aligned_names(
    sentence_section_names: Sequence[str],
) -> tuple[list[int], list[str]]:
    """Convert sentence-aligned headings into contiguous original sections.

    A new section begins whenever the heading changes. This preserves the
    document's section order even if the same heading appears again later.
    """
    if not sentence_section_names:
        return [], []

    section_ids: list[int] = []
    section_names: list[str] = []
    previous: str | None = None
    current_id = -1
    for raw_name in sentence_section_names:
        name = str(raw_name).strip() or "untitled"
        if previous is None or name != previous:
            current_id += 1
            section_names.append(name)
            previous = name
        section_ids.append(current_id)
    return section_ids, section_names


def flatten_sections(
    sections: Sequence[Any],
    section_names: Sequence[str] | None = None,
) -> tuple[list[str], list[int], list[str]]:
    """Flatten common nested section layouts while retaining section identity."""
    sentences: list[str] = []
    section_ids: list[int] = []
    names: list[str] = []

    for raw_index, section in enumerate(sections):
        if isinstance(section, Mapping):
            raw_sentences = section.get("sentences", section.get("text", []))
            name = str(
                section.get("heading")
                or section.get("title")
                or f"section-{raw_index}"
            )
        else:
            raw_sentences = section
            if section_names is not None and raw_index < len(section_names):
                name = str(section_names[raw_index])
            else:
                name = f"section-{raw_index}"

        if isinstance(raw_sentences, str):
            raw_sentences = [raw_sentences]
        if not isinstance(raw_sentences, Sequence):
            raise ValueError(f"section {raw_index} has no sentence sequence")

        clean_sentences = [str(sentence).strip() for sentence in raw_sentences]
        clean_sentences = [sentence for sentence in clean_sentences if sentence]
        if not clean_sentences:
            continue

        normalized_id = len(names)
        names.append(name.strip() or f"section-{raw_index}")
        sentences.extend(clean_sentences)
        section_ids.extend([normalized_id] * len(clean_sentences))

    return sentences, section_ids, names


def build_section_hyperedges(
    document: Mapping[str, Any],
    *,
    min_hyperedge_size: int = DEFAULT_MIN_HYPEREDGE_SIZE,
    max_hyperedge_size: int = DEFAULT_MAX_HYPEREDGE_SIZE,
) -> tuple[list[str], list[int], list[SectionHyperedge]]:
    """Read original section structure and construct retained hyperedges.

    Supported layouts are nested ``sections`` or sentence-aligned
    ``article_text`` plus ``section_ids``/``section_names``. If the document
    does not expose section membership, the function raises instead of
    inventing a structure not present in the source article.
    """
    raw_sections = document.get("sections")
    if isinstance(raw_sections, Sequence) and not isinstance(raw_sections, str):
        sentences, section_ids, names = flatten_sections(
            raw_sections,
            document.get("section_names"),
        )
    else:
        raw_sentences = document.get("article_text", document.get("sentences"))
        if not isinstance(raw_sentences, Sequence) or isinstance(raw_sentences, str):
            raise ValueError("document must provide article_text/sentences or sections")
        sentences = [str(sentence).strip() for sentence in raw_sentences]
        if any(not sentence for sentence in sentences):
            raise ValueError("document contains empty sentences")

        raw_ids = document.get("section_ids")
        raw_names = document.get("section_names")
        if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, str):
            if len(raw_ids) != len(sentences):
                raise ValueError("section_ids must be sentence-aligned")
            section_ids = [int(section_id) for section_id in raw_ids]
            names = list(raw_names) if isinstance(raw_names, Sequence) else []
        elif isinstance(raw_names, Sequence) and not isinstance(raw_names, str):
            if len(raw_names) != len(sentences):
                raise ValueError("section_names must be sentence-aligned")
            section_ids, names = section_ids_from_aligned_names(raw_names)
        else:
            raise ValueError("original sentence-to-section membership is required")

    hyperedges = section_hyperedges_from_ids(
        section_ids,
        names,
        min_hyperedge_size=min_hyperedge_size,
        max_hyperedge_size=max_hyperedge_size,
    )
    return sentences, section_ids, hyperedges


def section_incidence_matrix(
    sentence_count: int,
    hyperedges: Sequence[SectionHyperedge],
) -> np.ndarray:
    """Create the binary sentence-by-section-hyperedge incidence matrix."""
    if sentence_count < 0:
        raise ValueError("sentence_count must be non-negative")
    incidence = np.zeros((sentence_count, len(hyperedges)), dtype=np.int8)
    for hyperedge_index, hyperedge in enumerate(hyperedges):
        for sentence_node in hyperedge.sentence_nodes:
            if not 0 <= sentence_node < sentence_count:
                raise ValueError("hyperedge contains an invalid sentence node")
            incidence[sentence_node, hyperedge_index] = 1
    return incidence

