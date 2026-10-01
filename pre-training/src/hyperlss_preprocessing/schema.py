from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CleanDocument:
    document_id: str
    sentences: list[str]
    section_ids: list[int]
    section_names: list[str]
    reference_sentences: list[str]
    content_tokens: list[list[str]]
    sentence_token_counts: list[int]
    source_token_count: int
    oracle_labels: list[int]
    oracle_indices: list[int]
    oracle_objective: float

    def validate(self) -> None:
        n = len(self.sentences)
        if not n:
            raise ValueError("document contains no sentences")
        if len(self.section_ids) != n or len(self.content_tokens) != n:
            raise ValueError("sentence-aligned fields have different lengths")
        if len(self.sentence_token_counts) != n:
            raise ValueError("sentence_token_counts is not sentence-aligned")
        if len(self.oracle_labels) != n:
            raise ValueError("oracle_labels is not sentence-aligned")
        if not self.reference_sentences:
            raise ValueError("reference summary is empty")
        if self.section_ids and max(self.section_ids) >= len(self.section_names):
            raise ValueError("section id points outside section_names")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CleanDocument":
        value = cls(**raw)
        value.validate()
        return value


@dataclass(frozen=True)
class Rejection:
    document_id: str
    reason: str
    detail: str = ""
