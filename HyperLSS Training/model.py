from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from config import HyperLSSConfig


def segment_softmax(scores: torch.Tensor, groups: torch.Tensor, group_count: int) -> torch.Tensor:
    """Numerically stable softmax over arbitrary integer groups."""
    if scores.numel() == 0:
        return scores
    maxima = torch.full(
        (group_count,), -torch.inf, device=scores.device, dtype=scores.dtype
    )
    maxima.scatter_reduce_(0, groups, scores, reduce="amax", include_self=True)
    exponentials = torch.exp(scores - maxima[groups])
    denominators = torch.zeros(
        group_count, device=scores.device, dtype=scores.dtype
    )
    denominators.index_add_(0, groups, exponentials)
    return exponentials / denominators[groups].clamp_min(1e-12)


class RelationAwareGATLayer(nn.Module):
    """Equations (14)-(20): relation-wise attention with edge priors."""

    def __init__(self, config: HyperLSSConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.heads = config.hetero_heads
        self.head_dim = config.hidden_dim // config.hetero_heads
        self.relations = config.relation_count
        self.weight = nn.Parameter(
            torch.empty(self.relations, self.heads, self.head_dim, self.hidden_dim)
        )
        self.attention = nn.Parameter(
            torch.empty(self.relations, self.heads, 2 * self.head_dim)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attention)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        node_count = x.shape[0]
        source, target = edge_index[0], edge_index[1]
        head_outputs = [x.new_zeros((node_count, self.head_dim)) for _ in range(self.heads)]

        for relation in range(self.relations):
            mask = edge_type == relation
            if not torch.any(mask):
                continue
            relation_source = source[mask]
            relation_target = target[mask]
            relation_weight = edge_weight[mask]
            for head in range(self.heads):
                matrix = self.weight[relation, head]
                transformed_source = F.linear(x[relation_source], matrix)
                transformed_target = F.linear(x[relation_target], matrix)
                pair = torch.cat((transformed_target, transformed_source), dim=-1)
                logits = F.leaky_relu(
                    (pair * self.attention[relation, head]).sum(dim=-1)
                    + relation_weight,
                    negative_slope=0.2,
                )
                # Softmax is separate for every relation; target IDs form groups.
                coefficients = segment_softmax(logits, relation_target, node_count)
                messages = coefficients.unsqueeze(-1) * transformed_source
                head_outputs[head].index_add_(0, relation_target, messages)

        output = torch.cat([F.elu(value) for value in head_outputs], dim=-1)
        return self.dropout(output)


class MultiHeadHypergraphAttention(nn.Module):
    """Two-stage sentence -> hyperedge -> sentence multi-head attention.

    The paper specifies this two-step MH-HGA operation but does not expand its
    internal score equations. This implementation uses additive attention at
    both incidence directions while keeping the stated dimensions and fixed
    incidence structure.
    """

    def __init__(self, config: HyperLSSConfig) -> None:
        super().__init__()
        self.heads = config.hypergraph_heads
        self.head_dim = config.hidden_dim // config.hypergraph_heads
        d = config.hidden_dim
        self.sentence_to_edge = nn.Parameter(torch.empty(self.heads, self.head_dim, d))
        self.sentence_attention = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.node_query = nn.Parameter(torch.empty(self.heads, self.head_dim, d))
        self.edge_value = nn.Parameter(
            torch.empty(self.heads, self.head_dim, self.head_dim)
        )
        self.edge_attention = nn.Parameter(torch.empty(self.heads, 2 * self.head_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        sentences: torch.Tensor,
        incidence_sentence: torch.Tensor,
        incidence_hyperedge: torch.Tensor,
    ) -> torch.Tensor:
        sentence_count = sentences.shape[0]
        if incidence_hyperedge.numel() == 0:
            return sentences.new_zeros(sentences.shape)
        hyperedge_count = int(incidence_hyperedge.max().item()) + 1
        outputs: list[torch.Tensor] = []

        for head in range(self.heads):
            projected_sentences = F.linear(sentences, self.sentence_to_edge[head])
            node_logits = F.leaky_relu(
                (projected_sentences[incidence_sentence] * self.sentence_attention[head]).sum(-1),
                negative_slope=0.2,
            )
            node_coefficients = segment_softmax(
                node_logits, incidence_hyperedge, hyperedge_count
            )
            hyperedges = sentences.new_zeros((hyperedge_count, self.head_dim))
            hyperedges.index_add_(
                0,
                incidence_hyperedge,
                node_coefficients.unsqueeze(-1) * projected_sentences[incidence_sentence],
            )

            node_queries = F.linear(sentences, self.node_query[head])
            edge_values = F.linear(hyperedges, self.edge_value[head])
            pairs = torch.cat(
                (
                    node_queries[incidence_sentence],
                    edge_values[incidence_hyperedge],
                ),
                dim=-1,
            )
            edge_logits = F.leaky_relu(
                (pairs * self.edge_attention[head]).sum(-1), negative_slope=0.2
            )
            edge_coefficients = segment_softmax(
                edge_logits, incidence_sentence, sentence_count
            )
            output = sentences.new_zeros((sentence_count, self.head_dim))
            output.index_add_(
                0,
                incidence_sentence,
                edge_coefficients.unsqueeze(-1) * edge_values[incidence_hyperedge],
            )
            outputs.append(output)
        return torch.cat(outputs, dim=-1)


class HypergraphTransformerLayer(nn.Module):
    """Equations (23)-(26): MH-HGA, residual/LN, FFN, residual/LN."""

    def __init__(self, config: HyperLSSConfig) -> None:
        super().__init__()
        self.attention = MultiHeadHypergraphAttention(config)
        self.dropout = nn.Dropout(config.dropout)
        self.attention_norm = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.ReLU(),
            nn.Linear(config.ffn_dim, config.hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        sentences: torch.Tensor,
        incidence_sentence: torch.Tensor,
        incidence_hyperedge: torch.Tensor,
    ) -> torch.Tensor:
        attention = self.attention(
            sentences, incidence_sentence, incidence_hyperedge
        )
        normalized = self.attention_norm(sentences + self.dropout(attention))
        return self.ffn_norm(normalized + self.dropout(self.ffn(normalized)))


class GatedFusion(nn.Module):
    """Equations (27)-(28): feature-wise gated fusion."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, heterogeneous: torch.Tensor, hypergraph: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat((heterogeneous, hypergraph), dim=-1)))
        return gate * heterogeneous + (1.0 - gate) * hypergraph


class HyperLSS(nn.Module):
    """Two-channel, two-layer HyperLSS with intermediate feedback."""

    def __init__(self, config: HyperLSSConfig | None = None) -> None:
        super().__init__()
        self.config = config or HyperLSSConfig()
        self.config.validate()
        c = self.config
        self.sentence_projection = nn.Linear(c.sentence_input_dim, c.hidden_dim)
        self.word_projection = nn.Linear(c.word_input_dim, c.hidden_dim)
        self.heterogeneous_layers = nn.ModuleList(
            [RelationAwareGATLayer(c), RelationAwareGATLayer(c)]
        )
        self.hypergraph_layers = nn.ModuleList(
            [HypergraphTransformerLayer(c), HypergraphTransformerLayer(c)]
        )
        self.fusions = nn.ModuleList([GatedFusion(c.hidden_dim), GatedFusion(c.hidden_dim)])
        self.classifier_hidden = nn.Linear(c.hidden_dim, c.classifier_dim)
        self.classifier_dropout = nn.Dropout(c.dropout)
        self.classifier_output = nn.Linear(c.classifier_dim, 1)

    def forward(self, batch: Any) -> torch.Tensor:
        sentence_count = batch.sentence_features.shape[0]
        sentence_initial = self.sentence_projection(batch.sentence_features)
        word_initial = self.word_projection(batch.word_features)

        first_nodes = torch.cat((sentence_initial, word_initial), dim=0)
        first_heterogeneous_nodes = self.heterogeneous_layers[0](
            first_nodes, batch.edge_index, batch.edge_type, batch.edge_weight
        )
        first_heterogeneous_sentences = first_heterogeneous_nodes[:sentence_count]
        first_words = first_heterogeneous_nodes[sentence_count:]
        first_hypergraph_sentences = self.hypergraph_layers[0](
            sentence_initial, batch.incidence_sentence, batch.incidence_hyperedge
        )
        first_fused = self.fusions[0](
            first_heterogeneous_sentences, first_hypergraph_sentences
        )

        # Only sentence representations cross channels; word states remain in
        # the heterogeneous channel exactly as described in the paper.
        second_nodes = torch.cat((first_fused, first_words), dim=0)
        second_heterogeneous_nodes = self.heterogeneous_layers[1](
            second_nodes, batch.edge_index, batch.edge_type, batch.edge_weight
        )
        second_heterogeneous_sentences = second_heterogeneous_nodes[:sentence_count]
        second_hypergraph_sentences = self.hypergraph_layers[1](
            first_fused, batch.incidence_sentence, batch.incidence_hyperedge
        )
        final_sentences = self.fusions[1](
            second_heterogeneous_sentences, second_hypergraph_sentences
        )

        hidden = F.relu(self.classifier_hidden(final_sentences))
        logits = self.classifier_output(self.classifier_dropout(hidden)).squeeze(-1)
        return logits

