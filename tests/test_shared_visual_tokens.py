"""Invariants for the single visual projection/merge point.

Runs on CPU with plain torch: no LAVIS, no transformers, no GPU.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_encoders.shared_visual_tokens import (  # noqa: E402
    SharedVisualTokenProjector,
    SharedVisualTokens,
    validate_shared_visual_tokens,
)

# biovil is already at visual_dim; the others need alignment.
STREAM_DIMS = {"biovil": 1408, "pubmedclip": 768, "swin": 1024}
VISUAL_DIM = 1408


def make_streams(batch=3):
    return {
        "biovil": torch.randn(batch, 196, 1408),
        "pubmedclip": torch.randn(batch, 50, 768),
        "swin": torch.randn(batch, 49, 1024),
    }


class TestMergeShape(unittest.TestCase):
    def test_shared_tokens_shape_and_dim(self):
        projector = SharedVisualTokenProjector(STREAM_DIMS, VISUAL_DIM)
        shared = projector(make_streams(batch=3))
        self.assertEqual(shared.tokens.ndim, 3)
        self.assertEqual(shared.tokens.shape[0], 3)
        self.assertEqual(shared.tokens.shape[-1], VISUAL_DIM)
        self.assertEqual(shared.tokens.shape[1], 196 + 50 + 49)

    def test_spans_are_exact_contiguous_and_complete(self):
        projector = SharedVisualTokenProjector(STREAM_DIMS, VISUAL_DIM)
        shared = projector(make_streams())
        self.assertEqual(shared.spans["biovil"], slice(0, 196))
        self.assertEqual(shared.spans["pubmedclip"], slice(196, 246))
        self.assertEqual(shared.spans["swin"], slice(246, 295))
        covered = sum(s.stop - s.start for s in shared.spans.values())
        self.assertEqual(covered, shared.tokens.shape[1])
        validate_shared_visual_tokens(shared, VISUAL_DIM)

    def test_concat_order_is_canonical_not_insertion_order(self):
        """Shuffled construction and input order must not move a stream's tokens."""
        shuffled_dims = {"swin": 1024, "biovil": 1408, "pubmedclip": 768}
        projector = SharedVisualTokenProjector(shuffled_dims, VISUAL_DIM)
        self.assertEqual(projector.stream_names, ("biovil", "pubmedclip", "swin"))

        streams = make_streams()
        shuffled_streams = {k: streams[k] for k in ("swin", "pubmedclip", "biovil")}
        shared = projector(shuffled_streams)
        self.assertEqual(shared.spans["biovil"], slice(0, 196))
        self.assertEqual(shared.spans["swin"], slice(246, 295))

    def test_identity_projection_adds_no_parameters(self):
        projector = SharedVisualTokenProjector({"biovil": 1408}, VISUAL_DIM)
        self.assertEqual(sum(p.numel() for p in projector.parameters()), 0)


class TestBothBranchesSeeTheSameTensor(unittest.TestCase):
    def test_stream_view_is_a_slice_of_the_shared_tensor(self):
        """MHCAC's per-encoder view must come from the merged tensor, not a re-encode."""
        projector = SharedVisualTokenProjector(STREAM_DIMS, VISUAL_DIM)
        shared = projector(make_streams())
        for name, span in shared.spans.items():
            torch.testing.assert_close(shared.stream(name), shared.tokens[:, span, :])

    def test_single_forward_feeds_both_branches(self):
        """One projector call produces the object both branches consume."""
        projector = SharedVisualTokenProjector(STREAM_DIMS, VISUAL_DIM)
        shared = projector(make_streams())
        metaformer_input = shared.tokens
        mhcac_input = shared.tokens
        self.assertIs(metaformer_input, mhcac_input)


class TestMasking(unittest.TestCase):
    def test_masking_one_encoder_leaves_the_others_bit_identical(self):
        projector = SharedVisualTokenProjector(STREAM_DIMS, VISUAL_DIM)
        shared = projector(make_streams())
        ablated = shared.without("pubmedclip")

        self.assertTrue(torch.all(ablated.stream("pubmedclip") == 0))
        torch.testing.assert_close(ablated.stream("biovil"), shared.stream("biovil"))
        torch.testing.assert_close(ablated.stream("swin"), shared.stream("swin"))
        # Ablation must not change the token axis, only its contents.
        self.assertEqual(ablated.tokens.shape, shared.tokens.shape)
        self.assertEqual(ablated.spans, shared.spans)

    def test_stream_mask_selects_exactly_that_span(self):
        projector = SharedVisualTokenProjector(STREAM_DIMS, VISUAL_DIM)
        shared = projector(make_streams())
        mask = shared.stream_mask("swin")
        self.assertEqual(int(mask.sum()), 49)
        self.assertTrue(torch.all(mask[246:295]))
        self.assertFalse(torch.any(mask[:246]))

    def test_unknown_stream_raises(self):
        projector = SharedVisualTokenProjector(STREAM_DIMS, VISUAL_DIM)
        shared = projector(make_streams())
        with self.assertRaises(KeyError):
            shared.stream("raddino")
        with self.assertRaises(KeyError):
            shared.without("raddino")


class TestGradientRouting(unittest.TestCase):
    def test_gradient_reaches_the_matching_projection_only(self):
        projector = SharedVisualTokenProjector(STREAM_DIMS, VISUAL_DIM)
        shared = projector(make_streams())
        # Backprop through the swin span only.
        shared.stream("swin").sum().backward()

        swin_grad = projector.projections["swin"].weight.grad
        self.assertIsNotNone(swin_grad)
        self.assertGreater(float(swin_grad.abs().sum()), 0.0)
        # torch.cat backward allocates a grad tensor for every input, so the
        # other projections get a zero grad rather than None. Zero is the real
        # invariant: no gradient crosses from one encoder's span to another's.
        other_grad = projector.projections["pubmedclip"].weight.grad
        self.assertEqual(float(other_grad.abs().sum()), 0.0)

    def test_gradient_flows_to_every_projection_from_the_merged_tensor(self):
        projector = SharedVisualTokenProjector(STREAM_DIMS, VISUAL_DIM)
        shared = projector(make_streams())
        shared.tokens.sum().backward()
        for name in ("pubmedclip", "swin"):
            grad = projector.projections[name].weight.grad
            self.assertIsNotNone(grad, f"no gradient reached {name}")
            self.assertGreater(float(grad.abs().sum()), 0.0)


class TestValidationFailsClosed(unittest.TestCase):
    def test_overlapping_spans_rejected(self):
        shared = SharedVisualTokens(
            tokens=torch.randn(2, 10, VISUAL_DIM),
            spans={"biovil": slice(0, 6), "swin": slice(4, 10)},
        )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_shared_visual_tokens(shared, VISUAL_DIM)

    def test_incomplete_coverage_rejected(self):
        shared = SharedVisualTokens(
            tokens=torch.randn(2, 10, VISUAL_DIM),
            spans={"biovil": slice(0, 4)},
        )
        with self.assertRaisesRegex(ValueError, "spans"):
            validate_shared_visual_tokens(shared, VISUAL_DIM)

    def test_gap_between_spans_rejected(self):
        shared = SharedVisualTokens(
            tokens=torch.randn(2, 10, VISUAL_DIM),
            spans={"biovil": slice(0, 4), "swin": slice(6, 10)},
        )
        with self.assertRaisesRegex(ValueError, "unclaimed"):
            validate_shared_visual_tokens(shared, VISUAL_DIM)

    def test_wrong_visual_dim_rejected(self):
        shared = SharedVisualTokens(
            tokens=torch.randn(2, 10, 768),
            spans={"biovil": slice(0, 10)},
        )
        with self.assertRaisesRegex(ValueError, "configured visual_dim"):
            validate_shared_visual_tokens(shared, VISUAL_DIM)

    def test_two_dimensional_input_rejected(self):
        shared = SharedVisualTokens(
            tokens=torch.randn(10, VISUAL_DIM), spans={"biovil": slice(0, 10)}
        )
        with self.assertRaisesRegex(ValueError, r"\[B, N, D\]"):
            validate_shared_visual_tokens(shared, VISUAL_DIM)

    def test_missing_stream_at_forward_rejected(self):
        projector = SharedVisualTokenProjector(STREAM_DIMS, VISUAL_DIM)
        streams = make_streams()
        del streams["swin"]
        with self.assertRaisesRegex(ValueError, "missing encoder stream"):
            projector(streams)

    def test_unexpected_stream_at_forward_rejected(self):
        projector = SharedVisualTokenProjector(STREAM_DIMS, VISUAL_DIM)
        streams = make_streams()
        streams["raddino"] = torch.randn(3, 49, 768)
        with self.assertRaisesRegex(ValueError, "not built for"):
            projector(streams)

    def test_batch_mismatch_between_streams_rejected(self):
        projector = SharedVisualTokenProjector(STREAM_DIMS, VISUAL_DIM)
        streams = make_streams(batch=3)
        streams["swin"] = torch.randn(2, 49, 1024)
        with self.assertRaisesRegex(ValueError, "batch"):
            projector(streams)

    def test_unknown_stream_name_at_construction_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown encoder stream"):
            SharedVisualTokenProjector({"resnet": 512}, VISUAL_DIM)


if __name__ == "__main__":
    unittest.main()
