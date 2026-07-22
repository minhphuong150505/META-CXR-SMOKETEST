"""Unit tests for ViewFusionModule. Run with `python -m tests.test_view_fusion`.

No GPU and no dataset required.
"""
import torch

from mhcac.view_fusion import ViewFusionModule


B, P, D = 3, 7, 32


def make_module(**kw):
    torch.manual_seed(0)
    return ViewFusionModule(dim=D, num_heads=4, ffn_ratio=2, **kw)


def test_zero_init_identity():
    """Checklist 2: at step 0 the module is an exact identity on the anchor."""
    m = make_module(p_view_drop=0.0).eval()
    anchor = torch.randn(B, P, D)
    aux = torch.randn(B, 2, P, D)
    aux_mask = torch.ones(B, 2, dtype=torch.bool)
    out = m(anchor, aux, aux_mask,
            anchor_view_id=torch.tensor([0, 1, 2]),
            aux_view_ids=torch.tensor([[1, 2], [0, 3], [3, 1]]))
    max_diff = (out - anchor).abs().max().item()
    assert max_diff < 1e-6, max_diff
    print(f"ok: zero-init identity (max diff {max_diff:.2e})")


def test_no_aux_returns_anchor():
    """Checklist 3: an all-False aux_mask returns the anchor, no NaN."""
    m = make_module(p_view_drop=0.0).eval()
    # Break the zero init so the test proves the gate, not the init.
    for blk in m.blocks:
        torch.nn.init.normal_(blk.w_o.weight, std=0.5)
        torch.nn.init.normal_(blk.w_o.bias, std=0.5)
        torch.nn.init.normal_(blk.ffn[-1].weight, std=0.5)
        torch.nn.init.normal_(blk.ffn[-1].bias, std=0.5)

    anchor = torch.randn(B, P, D)
    aux = torch.randn(B, 2, P, D)
    aux_mask = torch.zeros(B, 2, dtype=torch.bool)
    out = m(anchor, aux, aux_mask)
    assert torch.isfinite(out).all(), "non-finite output for the n=0 case"
    assert torch.allclose(out, anchor, atol=1e-6), (out - anchor).abs().max()

    # Mixed batch: only row 1 has an auxiliary view.
    aux_mask = torch.tensor([[False, False], [True, False], [False, False]])
    out = m(anchor, aux, aux_mask)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[0], anchor[0], atol=1e-6)
    assert torch.allclose(out[2], anchor[2], atol=1e-6)
    assert not torch.allclose(out[1], anchor[1], atol=1e-6), \
        "row with an auxiliary view should have been modified"
    print("ok: n=0 gate returns anchor exactly, finite, batch-dense")


def test_shapes():
    """Checklist 4: [B,P,D] in -> [B,P,D] out for N in {0,1,3} and mixed counts."""
    m = make_module(p_view_drop=0.0).eval()
    anchor = torch.randn(B, P, D)
    for n in (0, 1, 3):
        aux = torch.randn(B, n, P, D)
        aux_mask = torch.ones(B, n, dtype=torch.bool)
        out = m(anchor, aux, aux_mask)
        assert out.shape == (B, P, D), (n, out.shape)

    # Per-sample counts 2 / 1 / 0, padded to N_max = 2.
    aux = torch.randn(B, 2, P, D)
    aux_mask = torch.tensor([[True, True], [True, False], [False, False]])
    out = m(anchor, aux, aux_mask)
    assert out.shape == (B, P, D)
    assert torch.isfinite(out).all()
    print("ok: shape contract holds for N in {0,1,3} and ragged counts")


def test_gradient_reaches_kv():
    """Gradient flows into the fusion weights despite the detached aux features.

    At exactly step 0 the zero-init W_O makes dL/dW_K and dL/dW_V vanish -- W_O
    is the only weight in the attention branch that moves. Once W_O is nonzero
    (i.e. from step 1 on) K/V start learning, so both facts are asserted.
    """
    anchor = torch.randn(B, P, D)
    aux = torch.randn(B, 2, P, D).detach()  # frozen-encoder output, no_grad
    aux_mask = torch.ones(B, 2, dtype=torch.bool)

    m = make_module(p_view_drop=0.0).train()
    m(anchor, aux, aux_mask).sum().backward()
    g_o = m.blocks[0].w_o.weight.grad
    assert g_o is not None and g_o.abs().sum() > 0, "W_O received no gradient at init"

    m = make_module(p_view_drop=0.0).train()
    torch.nn.init.normal_(m.blocks[0].w_o.weight, std=0.1)  # simulate step >= 1
    m(anchor, aux, aux_mask).sum().backward()
    for name in ("w_k", "w_v"):
        g = getattr(m.blocks[0], name).weight.grad
        assert g is not None and g.abs().sum() > 0, f"{name} received no gradient"
    print("ok: W_O learns at step 0; W_K / W_V learn once W_O is nonzero")


def test_view_dropout_train_only():
    """Checklist 6: view dropout is active in train() and inert in eval().

    p_view_drop=1.0 makes this deterministic: under train() every auxiliary is
    dropped, so the n=0 gate returns the anchor; under eval() the views survive
    and the output must differ.
    """
    m = make_module(p_view_drop=1.0, dropout=0.0)
    for blk in m.blocks:  # break the zero init so fusion has a visible effect
        torch.nn.init.normal_(blk.w_o.weight, std=0.5)
    anchor = torch.randn(B, P, D)
    aux = torch.randn(B, 4, P, D)
    aux_mask = torch.ones(B, 4, dtype=torch.bool)

    m.train()
    out_train = m(anchor, aux, aux_mask)
    assert torch.allclose(out_train, anchor, atol=1e-6), \
        "train() with p_view_drop=1.0 should drop every view and return the anchor"

    m.eval()
    out_eval = m(anchor, aux, aux_mask)
    assert not torch.allclose(out_eval, anchor, atol=1e-6), \
        "eval() must ignore view dropout and actually fuse"
    print("ok: view dropout active under train(), inert under eval()")


if __name__ == "__main__":
    test_zero_init_identity()
    test_no_aux_returns_anchor()
    test_shapes()
    test_gradient_reaches_kv()
    test_view_dropout_train_only()
    print("\nall view-fusion tests passed")
