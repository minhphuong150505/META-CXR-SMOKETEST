import torch
from torch import nn
from torch.nn import functional as F


# Additive mask value for padded auxiliary tokens. Finite (not -inf) so a row
# that is entirely padding still produces a valid softmax instead of NaN.
MASK_NEG = -1e4


class ViewFusionBlock(nn.Module):
    """One pre-norm cross-attention block with zero-init residual gating.

        h   = anchor + Dropout(W_O . MHA(LN_q(anchor + view_emb),
                                         LN_kv(aux + view_emb), mask))
        out = h      + Dropout(FFN(LN_f(h)))

    ``W_O`` and the FFN output ``Linear`` are zero-initialised, so at step 0 the
    block is an exact identity on ``anchor``. Pre-norm keeps the residual stream
    unscaled: the fused tensors are raw pre-projection encoder outputs, and the
    downstream pretrained projections were fitted against that scale.
    """

    def __init__(self, dim, num_heads, ffn_ratio, dropout):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)

        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_o = nn.Linear(dim, dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_ratio, dim),
        )
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.w_o.weight)
        nn.init.zeros_(self.w_o.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def _split_heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, anchor, anchor_view_emb, aux_tokens, attn_bias, gate):
        """anchor [B, P, D], anchor_view_emb [B, 1, D] or None,
        aux_tokens [B, N*P, D], attn_bias [B, 1, 1, N*P],
        gate [B, 1, 1] float, 0 for studies with no auxiliary view."""
        B, P, D = anchor.shape

        query = anchor if anchor_view_emb is None else anchor + anchor_view_emb
        q = self._split_heads(self.w_q(self.norm_q(query)))
        kv = self.norm_kv(aux_tokens)
        k = self._split_heads(self.w_k(kv))
        v = self._split_heads(self.w_v(kv))

        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = attn + attn_bias
        attn = F.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, P, D)

        h = anchor + self.dropout(self.w_o(ctx)) * gate
        return h + self.dropout(self.ffn(self.norm_ffn(h))) * gate


class ViewFusionModule(nn.Module):
    """Fuse auxiliary-view tokens into the anchor's token sequence.

    Shape contract: ``[B, P, D]`` in, ``[B, P, D]`` out, so the downstream MHCAC
    and Q-Former branches need no change. ``dim`` is per-stream (biovil 1408,
    PubMedCLIP 768 and SwinV2 its own embed_dim), so one module is
    instantiated per enabled encoder.
    """

    def __init__(self, dim, num_heads=8, ffn_ratio=4, num_view_types=4,
                 dropout=0.1, num_blocks=1, p_view_drop=0.2):
        super().__init__()
        self.p_view_drop = p_view_drop
        self.view_emb = nn.Embedding(num_view_types, dim)
        nn.init.normal_(self.view_emb.weight, std=0.02)
        self.blocks = nn.ModuleList([
            ViewFusionBlock(dim, num_heads, ffn_ratio, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, anchor, aux, aux_mask=None, anchor_view_id=None,
                aux_view_ids=None):
        """
        anchor:          [B, P, D]
        aux:             [B, N, P, D]   auxiliary views, padded to the batch N_max
        aux_mask:        [B, N] bool    True = real view, False = padding
        anchor_view_id:  [B]            view type of the anchor (0..3)
        aux_view_ids:    [B, N]         view type of each auxiliary
        returns:         [B, P, D]
        """
        if aux is None or aux.numel() == 0 or aux.shape[1] == 0:
            return anchor

        B, N, P, D = aux.shape
        device = anchor.device

        if aux_mask is None:
            aux_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        else:
            aux_mask = aux_mask.to(device=device, dtype=torch.bool)

        if self.training and self.p_view_drop > 0:
            keep = torch.rand(B, N, device=device) >= self.p_view_drop
            aux_mask = aux_mask & keep

        # View-type embeddings enter only the attention branch, never the
        # residual stream, so they are nullified along with W_O at init.
        anchor_view_emb = None
        if anchor_view_id is not None:
            anchor_view_emb = self.view_emb(
                anchor_view_id.to(device=device, dtype=torch.long)
            ).unsqueeze(1)

        aux_tokens = aux
        if aux_view_ids is not None:
            aux_tokens = aux_tokens + self.view_emb(
                aux_view_ids.to(device=device, dtype=torch.long)
            ).unsqueeze(2)
        aux_tokens = aux_tokens.reshape(B, N * P, D)

        # Studies with no auxiliary view: force slot 0 open so softmax sees a
        # valid row, then gate both residual branches to zero. Keeps the batch
        # dense and NaN-free without splitting it, and returns `anchor` exactly.
        has_aux = aux_mask.any(dim=1)
        open_mask = aux_mask.clone()
        open_mask[~has_aux, 0] = True

        token_mask = open_mask.repeat_interleave(P, dim=1)  # [B, N*P]
        attn_bias = torch.zeros(B, 1, 1, N * P, device=device, dtype=anchor.dtype)
        attn_bias = attn_bias.masked_fill(~token_mask[:, None, None, :], MASK_NEG)
        gate = has_aux.to(anchor.dtype).view(B, 1, 1)

        out = anchor
        for block in self.blocks:
            out = block(out, anchor_view_emb, aux_tokens, attn_bias, gate)
        return out
