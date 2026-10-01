from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HyperLSSConfig:
    sentence_input_dim: int = 384
    word_input_dim: int = 300
    hidden_dim: int = 256
    hetero_heads: int = 4
    hypergraph_heads: int = 8
    relation_count: int = 4
    ffn_dim: int = 1024
    classifier_dim: int = 128
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 4e-4
    global_batch_size: int = 64
    max_epochs: int = 20
    early_stopping_patience: int = 7

    def validate(self) -> None:
        if self.hidden_dim % self.hetero_heads:
            raise ValueError("hidden_dim must be divisible by hetero_heads")
        if self.hidden_dim % self.hypergraph_heads:
            raise ValueError("hidden_dim must be divisible by hypergraph_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def target_summary_sentences(dataset: str) -> int:
    values = {"pubmed": 8, "arxiv": 7}
    try:
        return values[dataset.lower()]
    except KeyError as exc:
        raise ValueError("dataset must be 'pubmed' or 'arxiv'") from exc

