from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessConfig:
    dataset: str
    target_summary_sentences: int
    max_source_tokens: int
    max_source_sentences: int

    min_source_tokens: int = 100
    min_source_sentences: int = 10
    invalid_character_ratio: float = 0.05
    max_sentence_tokens: int = 256
    max_sentence_truncation_ratio: float = 0.30

    sentence_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"

    @classmethod
    def for_dataset(cls, dataset: str) -> "PreprocessConfig":
        presets = {
            "pubmed": dict(
                target_summary_sentences=8,
                max_source_tokens=4000,
                max_source_sentences=400,
            ),
            "arxiv": dict(
                target_summary_sentences=7,
                max_source_tokens=5500,
                max_source_sentences=500,
            ),
        }
        if dataset not in presets:
            raise ValueError("dataset must be 'pubmed' or 'arxiv'")
        config = cls(dataset=dataset, **presets[dataset])
        config.validate()
        return config

    def validate(self) -> None:
        if self.dataset not in {"pubmed", "arxiv"}:
            raise ValueError("dataset must be 'pubmed' or 'arxiv'")
        if self.min_source_tokens >= self.max_source_tokens:
            raise ValueError("min_source_tokens must be below max_source_tokens")
        if self.min_source_sentences >= self.max_source_sentences:
            raise ValueError("min_source_sentences must be below max_source_sentences")
        if not 0.0 <= self.max_sentence_truncation_ratio < 1.0:
            raise ValueError("max_sentence_truncation_ratio must be in [0, 1)")
