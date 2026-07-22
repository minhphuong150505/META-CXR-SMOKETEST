"""Dependency-free FINDINGS parser for MIMIC-CXR text reports."""

from __future__ import annotations

import re


# Explicit names avoid treating short radiology sentences as section headers.
SECTION_ALIASES = {
    "findings": "findings",
    "finding": "findings",
    "findings and impression": "findings_impression",
    "findings/impression": "findings_impression",
    "impression": "impression",
    "conclusion": "impression",
    "conclusions": "impression",
    "examination": "examination",
    "exam": "examination",
    "procedure": "examination",
    "indication": "indication",
    "reason for examination": "indication",
    "clinical indication": "indication",
    "history": "history",
    "clinical history": "history",
    "technique": "technique",
    "comparison": "comparison",
    "comparisons": "comparison",
    "notification": "notification",
    "recommendation": "recommendation",
}
_SECTION_NAMES = sorted(SECTION_ALIASES, key=len, reverse=True)
SECTION_HEADER_RE = re.compile(
    r"^[ \t]*(?P<name>" + "|".join(re.escape(name) for name in _SECTION_NAMES)
    + r")[ \t]*:[ \t]*(?P<content>.*)$",
    re.IGNORECASE,
)
INLINE_TARGET_HEADER_RE = re.compile(
    r"(?<!^)(?<!/)(?<!AND )(?=(?:FINDINGS(?:\s+AND\s+IMPRESSION|/IMPRESSION)?|"
    r"IMPRESSION|CONCLUSIONS?)[ \t]*:)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[./'-][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
BOILERPLATE_LINES = {"final report", "preliminary report", "wet read"}


def _normalise_section_name(name: str) -> str:
    return SECTION_ALIASES[re.sub(r"\s+", " ", name.strip().lower())]


def _report_sections(report_text: str) -> tuple[dict[str, list[str]], list[str]]:
    """Parse common layouts while preventing metadata/target section bleed."""
    sections: dict[str, list[str]] = {}
    narrative: list[str] = []
    current: str | None = None
    text = report_text.replace("\r\n", "\n").replace("\r", "\n")

    for raw_line in text.split("\n"):
        # Some reports put multiple target headers on one physical line.
        for part in INLINE_TARGET_HEADER_RE.split(raw_line):
            line = part.strip()
            if not line:
                if current not in {"findings", "findings_impression", "impression"}:
                    current = None
                continue

            match = SECTION_HEADER_RE.match(line)
            if match:
                current = _normalise_section_name(match.group("name"))
                content = match.group("content").strip()
                if content:
                    sections.setdefault(current, []).append(content)
                continue

            normalised = re.sub(r"\s+", " ", line).strip(" :_").lower()
            if normalised in BOILERPLATE_LINES or not normalised:
                continue
            if current is None:
                narrative.append(line)
            else:
                sections.setdefault(current, []).append(line)

    return sections, narrative


def _narrative_after_comparison(comparison: str) -> str:
    """Conservatively recover an unlabelled body after comparison text."""
    comparison = comparison.strip()
    if not re.match(
        r"(?i)^(?:comparison|compared|reviewed|prior|previous|none(?: available)?|"
        r"no (?:prior|comparison))\b",
        comparison,
    ):
        return ""
    boundary = re.search(r"[.!?](?:\s+|$)", comparison)
    if boundary is None:
        return ""
    return comparison[boundary.end():].strip()


def extract_sections(report_text: str) -> tuple[str, str]:
    """Return FINDINGS and IMPRESSION without substituting one for another."""
    if not isinstance(report_text, str) or not report_text.strip():
        return "", ""
    sections, narrative = _report_sections(report_text)
    findings = " ".join(sections.get("findings", [])).strip()
    impression = " ".join(sections.get("impression", [])).strip()
    if not findings and narrative:
        findings = " ".join(narrative).strip()
    if not findings:
        findings = _narrative_after_comparison(" ".join(sections.get("comparison", [])))
    return findings, impression


def get_target_text(report_text: str) -> tuple[str, str, str]:
    """Return a FINDINGS-only target, impression for audit, and provenance."""
    if not isinstance(report_text, str) or not report_text.strip():
        return "", "", "EMPTY"

    sections, narrative = _report_sections(report_text)
    impression = " ".join(sections.get("impression", [])).strip()
    findings = " ".join(sections.get("findings", [])).strip()
    if findings:
        return findings, impression, "FINDINGS_TAG"

    combined = " ".join(sections.get("findings_impression", [])).strip()
    if combined:
        # A combined section cannot be separated reliably into the requested
        # FINDINGS-only target, so retain the study but mask generation.
        return "", combined, "FINDINGS_IMPRESSION_COMBINED"

    body = " ".join(narrative).strip()
    if body:
        return body, impression, "NARRATIVE_BODY"

    body = _narrative_after_comparison(" ".join(sections.get("comparison", [])))
    if body:
        return body, impression, "NARRATIVE_AFTER_COMPARISON"

    if impression:
        return "", impression, "IMPRESSION_ONLY"
    return "", "", "NO_FINDINGS"


def clean_report_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[\*\*.*?\*\*\]", "", text)
    text = re.sub(r"_{2,}", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()


def count_lexical_tokens(text: str) -> int:
    """Count deterministic word/punctuation tokens without model downloads."""
    if not isinstance(text, str):
        return 0
    return len(TOKEN_RE.findall(text))
