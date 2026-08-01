"""The single place where per-encoder features become one visual token sequence.

Before this module the two downstream branches were built from different
tensors: META-Former consumed a concatenation of the per-encoder Q-Former
projections, while MHCAC received the raw streams and re-projected them through
its own ``CrossModalEmbeddingAlignment``. Two independent projection/merge paths
meant the branches were never trained on the same visual representation.

``SharedVisualTokenProjector`` owns dimension alignment and the merge. Both
branches read the ``SharedVisualTokens.tokens`` it returns.

``spans`` is metadata only. It exists so per-encoder ablation, encoder-specific
heatmaps and selective masking can be expressed as slices/masks *over the shared
sequence* rather than by rebuilding a per-stream pipeline. Nothing downstream may
use a span to re-project a stream.

Depends on torch only, so the merge invariants are testable on a CPU box.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# Fixed concatenation order. Stable across runs and independent of dict
# insertion order, so a checkpoint's token positions keep their meaning.
CANONICAL_STREAM_ORDER = ("biovil", "pubmedclip", "swin")


@dataclass(frozen=True)
class SharedVisualTokens:
    """One visual token sequence plus the span each encoder occupies in it."""

    tokens: torch.Tensor  # [B, N_total, D]
    spans: dict[str, slice]

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def visual_dim(self) -> int:
        return int(self.tokens.shape[-1])

    def stream(self, name: str) -> torch.Tensor:
        """View of one encoder's tokens. A slice of the shared tensor, not a re-encode."""
        if name not in self.spans:
            raise KeyError(f"unknown stream {name!r}; have {sorted(self.spans)}")
        return self.tokens[:, self.spans[name], :]

    def stream_mask(self, name: str) -> torch.Tensor:
        """Bool mask over the token axis selecting one encoder. Shape [N_total]."""
        if name not in self.spans:
            raise KeyError(f"unknown stream {name!r}; have {sorted(self.spans)}")
        mask = torch.zeros(self.num_tokens, dtype=torch.bool, device=self.tokens.device)
        mask[self.spans[name]] = True
        return mask

    def select(self, *names: str) -> SharedVisualTokens:
        """Return only the requested encoder streams, with contiguous spans.

        Unlike :meth:`without`, this removes disabled tokens from the sequence.
        It is intended for paper-style inference ablations, where an inactive
        encoder must not remain visible to downstream projection biases,
        positional embeddings, or attention softmax normalization.
        """
        requested = {str(name) for name in names}
        if not requested:
            raise ValueError("select must retain at least one encoder stream")
        unknown = requested - set(self.spans)
        if unknown:
            raise KeyError(
                f"unknown stream(s) {sorted(unknown)}; have {sorted(self.spans)}"
            )

        selected = []
        spans = {}
        cursor = 0
        for name, span in sorted(self.spans.items(), key=lambda item: item[1].start):
            if name not in requested:
                continue
            stream = self.tokens[:, span, :]
            selected.append(stream)
            spans[name] = slice(cursor, cursor + stream.shape[1])
            cursor += stream.shape[1]
        return SharedVisualTokens(tokens=torch.cat(selected, dim=1), spans=spans)

    def without(self, *names: str) -> SharedVisualTokens:
        """Ablate encoders by zeroing their tokens in place of a per-stream rerun.

        Spans are preserved so the token axis keeps its length and meaning; only
        the named encoders' values are dropped.
        """
        tokens = self.tokens.clone()
        for name in names:
            if name not in self.spans:
                raise KeyError(f"unknown stream {name!r}; have {sorted(self.spans)}")
            tokens[:, self.spans[name], :] = 0.0
        return SharedVisualTokens(tokens=tokens, spans=dict(self.spans))


def validate_shared_visual_tokens(shared: SharedVisualTokens, visual_dim: int) -> None:
    """Fail closed on any merge that downstream code could silently misread."""
    tokens = shared.tokens
    if tokens.ndim != 3:
        raise ValueError(f"shared_visual_tokens must be [B, N, D]; got shape {tuple(tokens.shape)}")
    if tokens.shape[-1] != visual_dim:
        raise ValueError(
            f"shared_visual_tokens has dim {tokens.shape[-1]}, configured visual_dim is {visual_dim}"
        )
    if not shared.spans:
        raise ValueError("shared_visual_tokens carries no spans")

    ordered = sorted(shared.spans.items(), key=lambda item: item[1].start)
    covered = 0
    previous_stop = 0
    for name, span in ordered:
        if span.step not in (None, 1):
            raise ValueError(f"span for {name!r} must be contiguous; got step {span.step}")
        if span.start < previous_stop:
            raise ValueError(
                f"span for {name!r} overlaps the preceding stream "
                f"({span.start} < {previous_stop})"
            )
        if span.start != previous_stop:
            raise ValueError(
                f"spans leave tokens [{previous_stop}, {span.start}) unclaimed before {name!r}"
            )
        length = span.stop - span.start
        if length <= 0:
            raise ValueError(f"span for {name!r} is empty")
        covered += length
        previous_stop = span.stop

    if covered != tokens.shape[1]:
        raise ValueError(
            f"spans cover {covered} tokens but the sequence has {tokens.shape[1]}"
        )
    if previous_stop != tokens.shape[1]:
        raise ValueError(
            f"spans stop at {previous_stop} but the sequence has {tokens.shape[1]} tokens"
        )


class SharedVisualTokenProjector(nn.Module):
    """Aligns every enabled encoder to ``visual_dim`` and concatenates on the token axis.

    A stream already at ``visual_dim`` gets ``nn.Identity`` rather than a square
    Linear, so enabling this module adds no parameters to a stream that did not
    previously have a projection.
    """

    def __init__(self, stream_dims: dict[str, int], visual_dim: int = 1408):
        super().__init__()
        if not stream_dims:
            raise ValueError("at least one encoder stream must be enabled")
        unknown = set(stream_dims) - set(CANONICAL_STREAM_ORDER)
        if unknown:
            raise ValueError(
                f"unknown encoder stream(s) {sorted(unknown)}; "
                f"extend CANONICAL_STREAM_ORDER to add one"
            )
        self.visual_dim = int(visual_dim)
        # Ordered by the canonical order, not by the caller's dict.
        self.stream_names = tuple(
            name for name in CANONICAL_STREAM_ORDER if name in stream_dims
        )
        self.projections = nn.ModuleDict(
            {
                name: (
                    nn.Identity()
                    if int(stream_dims[name]) == self.visual_dim
                    else nn.Linear(int(stream_dims[name]), self.visual_dim)
                )
                for name in self.stream_names
            }
        )

    def forward(self, streams: dict[str, torch.Tensor]) -> SharedVisualTokens:
        missing = set(self.stream_names) - set(streams)
        if missing:
            raise ValueError(
                f"missing encoder stream(s) {sorted(missing)}; "
                f"this projector was built for {list(self.stream_names)}"
            )
        extra = set(streams) - set(self.stream_names)
        if extra:
            raise ValueError(
                f"received stream(s) {sorted(extra)} the projector was not built for; "
                f"expected exactly {list(self.stream_names)}"
            )

        projected: list[torch.Tensor] = []
        spans: dict[str, slice] = {}
        cursor = 0
        batch_size = None
        for name in self.stream_names:
            tensor = streams[name]
            if tensor.ndim != 3:
                raise ValueError(
                    f"stream {name!r} must be [B, P, D]; got shape {tuple(tensor.shape)}"
                )
            if batch_size is None:
                batch_size = int(tensor.shape[0])
            elif int(tensor.shape[0]) != batch_size:
                raise ValueError(
                    f"stream {name!r} has batch {tensor.shape[0]}, expected {batch_size}"
                )
            aligned = self.projections[name](tensor)
            spans[name] = slice(cursor, cursor + int(aligned.shape[1]))
            cursor += int(aligned.shape[1])
            projected.append(aligned)

        shared = SharedVisualTokens(tokens=torch.cat(projected, dim=1), spans=spans)
        validate_shared_visual_tokens(shared, self.visual_dim)
        return shared
