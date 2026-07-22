"""Synthetic CPU smoke tests for MIMIC report parsing and study sampling.

Run with ``python tests/test_mimic_data_pipeline.py``.  The test intentionally
loads dependency-light helper files directly, so it needs neither MIMIC data nor
the training environment's Torch/Pandas stack.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


parser = load_module("mimic_report_parser", "preporcessing/mimic_report_parser.py")
sampling = load_module("mimic_cxr_utils", "model/lavis/data/mimic_cxr_utils.py")


class ReportParserTest(unittest.TestCase):
    def test_explicit_findings_does_not_include_impression(self):
        findings, impression, method = parser.get_target_text(
            "FINDINGS: Heart size is normal.\nIMPRESSION: No acute disease."
        )
        self.assertEqual(findings, "Heart size is normal.")
        self.assertEqual(impression, "No acute disease.")
        self.assertEqual(method, "FINDINGS_TAG")
        self.assertNotIn("acute disease", findings)

    def test_impression_only_is_not_a_generation_target(self):
        findings, impression, method = parser.get_target_text(
            "FINAL REPORT\nIMPRESSION: Mild pulmonary edema."
        )
        self.assertEqual(findings, "")
        self.assertEqual(impression, "Mild pulmonary edema.")
        self.assertEqual(method, "IMPRESSION_ONLY")

    def test_combined_section_is_not_treated_as_pure_findings(self):
        findings, combined, method = parser.get_target_text(
            "FINDINGS/IMPRESSION: No acute cardiopulmonary process."
        )
        self.assertEqual(findings, "")
        self.assertEqual(combined, "No acute cardiopulmonary process.")
        self.assertEqual(method, "FINDINGS_IMPRESSION_COMBINED")

    def test_unlabelled_narrative_after_preamble_is_recovered(self):
        findings, _, method = parser.get_target_text(
            "INDICATION: cough\n\nCOMPARISON: None.\n\n"
            "The lungs are clear. Heart size is normal."
        )
        self.assertEqual(findings, "The lungs are clear. Heart size is normal.")
        self.assertEqual(method, "NARRATIVE_BODY")
        self.assertNotIn("cough", findings)

    def test_comparison_fallback_stops_before_impression(self):
        findings, impression, method = parser.get_target_text(
            "COMPARISON: Reviewed in comparison to prior. The lungs are clear.\n"
            "IMPRESSION: No acute disease."
        )
        self.assertEqual(findings, "The lungs are clear.")
        self.assertEqual(impression, "No acute disease.")
        self.assertEqual(method, "NARRATIVE_AFTER_COMPARISON")

    def test_inline_headers_and_token_count(self):
        findings, impression, _ = parser.get_target_text(
            "Findings: No focal opacity. Impression: Normal chest."
        )
        self.assertEqual(findings, "No focal opacity.")
        self.assertEqual(impression, "Normal chest.")
        self.assertEqual(parser.count_lexical_tokens(findings), 4)


class StudySamplingTest(unittest.TestCase):
    def test_anchor_priority_and_complementary_auxiliary(self):
        rows = [
            {"subject_id": 1, "study_id": 10, "ViewPosition": "AP"},
            {"subject_id": 1, "study_id": 10, "ViewPosition": "PA"},
            {"subject_id": 1, "study_id": 10, "ViewPosition": "PA"},
            {"subject_id": 1, "study_id": 10, "ViewPosition": "LATERAL"},
        ]
        [study] = sampling.build_study_index(rows, max_aux_views=1)
        self.assertEqual(study["anchor"], 1)
        # The repeated PA is skipped; AP has higher auxiliary priority than lateral.
        self.assertEqual(study["aux"], [0])
        self.assertEqual(study["anchor_view_id"], sampling.VIEW_ID_MAP["PA"])
        self.assertEqual(study["aux_view_ids"], [sampling.VIEW_ID_MAP["AP"]])

    def test_subject_is_part_of_study_identity(self):
        rows = [
            {"subject_id": 1, "study_id": 10, "ViewPosition": "PA"},
            {"subject_id": 2, "study_id": 10, "ViewPosition": "AP"},
        ]
        studies = sampling.build_study_index(rows)
        self.assertEqual(len(studies), 2)

    def test_at_most_one_auxiliary(self):
        with self.assertRaises(ValueError):
            sampling.build_study_index([], max_aux_views=2)


if __name__ == "__main__":
    unittest.main()
