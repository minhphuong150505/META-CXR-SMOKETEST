#!/usr/bin/env python3
"""Block notebooks carrying MIMIC-CXR data from entering Git.

MIMIC-CXR is PhysioNet credentialed-access data and the DUA forbids
redistribution. A notebook is the easiest way to leak it by accident: the source
looks harmless, and the executed *outputs* silently embed patient identifiers,
report text and `findings_clean` values. `.gitignore` protects the two known
notebooks, but a single `git add -f`, a rename, or a new notebook defeats it.

This checker is the backstop. It reads notebooks, reports violations, and --
importantly -- **never prints an identifier in full**. Printing the value in
order to warn about the value would put it in your terminal scrollback, your CI
log and your shell history.

Usage:

    python scripts/check_notebook_privacy.py --staged
    python scripts/check_notebook_privacy.py --path notebooks/foo.ipynb
    python scripts/check_notebook_privacy.py --all-tracked
    python scripts/check_notebook_privacy.py --sanitize in.ipynb --output out.ipynb

Exit code 0 = clean, 1 = violations found, 2 = usage/IO error.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

EXIT_OK, EXIT_VIOLATION, EXIT_ERROR = 0, 1, 2

# Identifier-shaped keys. Matched as JSON keys or as `key=`/`key:` in text.
IDENTIFIER_KEYS = ("subject_id", "study_id", "dicom_id", "patient_id")

# Dataset names. Their presence in *output* means real rows were probably shown.
MIMIC_MARKERS = ("MIMIC-CXR", "MIMIC-IV", "mimic-cxr", "mimic-iv", "mimic_cxr")

# Filesystem layout unique to MIMIC-CXR: files/p1X/pNNNNNNNN/sNNNNNNN/...
MIMIC_PATH = re.compile(r"\bp1\d/p\d{7,8}/s\d{7,8}\b|\bfiles/p1\d/p\d{7,8}\b")

# Private object storage.
PRIVATE_URI = re.compile(r"\b(?:gs|s3|azure|abfss)://[A-Za-z0-9._\-]+", re.IGNORECASE)

# Credentials and pre-signed URLs.
CREDENTIAL_PATTERNS = (
    ("aws_signed_url", re.compile(r"X-Amz-(?:Signature|Credential)=", re.IGNORECASE)),
    ("gcs_signed_url", re.compile(r"[?&]GoogleAccessId=|[?&]Signature=", re.IGNORECASE)),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{16,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}")),
    ("generic_secret", re.compile(
        r"\b(?:api[_-]?key|secret|password|token)\s*[=:]\s*[\"'][^\"'\s]{12,}[\"']",
        re.IGNORECASE,
    )),
)

# A MIMIC subject_id is 8 digits starting with 1; study_id is 8 digits starting
# with 5. Bare numbers are NOT flagged on shape alone -- tracked notebooks carry
# legitimate Kaggle datasetId/sourceId values in that exact range, and flagging
# those would train people to ignore this checker. A number counts only when it
# sits next to an identifier key or a MIMIC marker on the same line.
BARE_ID = re.compile(r"\b(?:1\d{7}|5\d{7})\b")
ID_KEY_NEARBY = re.compile(
    r"\b(?:subject_id|study_id|dicom_id|patient_id)\b|MIMIC", re.IGNORECASE
)


def redact(value: str) -> str:
    """`10000032` -> `10****32`. Enough to correlate, not enough to identify."""
    text = str(value)
    if len(text) <= 4:
        return "*" * len(text)
    keep = 2
    return f"{text[:keep]}{'*' * (len(text) - 2 * keep)}{text[-keep:]}"


@dataclass(frozen=True)
class Violation:
    path: str
    cell_index: int
    kind: str
    location: str
    redacted: str

    def format(self) -> str:
        return (
            f"{self.path}: cell {self.cell_index} [{self.location}] "
            f"{self.kind}: {self.redacted}"
        )


def _scan_text(text: str, path: str, cell_index: int, location: str) -> list[Violation]:
    """Find identifiers, MIMIC paths, private URIs and credentials in one blob."""
    found: list[Violation] = []

    for kind, pattern in CREDENTIAL_PATTERNS:
        for match in pattern.finditer(text):
            found.append(
                Violation(path, cell_index, kind, location, redact(match.group(0)))
            )

    for key in IDENTIFIER_KEYS:
        # A bare word is enough here: ``_scan_text`` runs on cell *outputs* only.
        # A DataFrame print renders column headers inline
        # ("   subject_id  study_id  ..."), so requiring `key:` or `key=` would
        # miss the single most common way this data leaks. Cell *source* is
        # scanned separately and deliberately does not flag identifier keys,
        # since `df["subject_id"]` in code is not a leak.
        if re.search(rf"\b{key}\b", text):
            found.append(
                Violation(path, cell_index, f"identifier_key:{key}", location, redact(key))
            )

    for match in MIMIC_PATH.finditer(text):
        found.append(
            Violation(path, cell_index, "mimic_data_path", location, redact(match.group(0)))
        )

    for match in PRIVATE_URI.finditer(text):
        found.append(
            Violation(path, cell_index, "private_storage_uri", location, redact(match.group(0)))
        )

    for marker in MIMIC_MARKERS:
        if marker in text:
            found.append(
                Violation(path, cell_index, "mimic_dataset_marker", location, redact(marker))
            )
            break

    # Identifier-shaped numbers count only when this output also carries an
    # identifier key or a MIMIC marker, so Kaggle datasetId/sourceId values in
    # unrelated output do not trip the checker. Context is evaluated across the
    # whole output, not per line: a DataFrame print puts the `subject_id`
    # header on one line and the values on the next.
    if ID_KEY_NEARBY.search(text):
        for match in BARE_ID.finditer(text):
            found.append(
                Violation(
                    path, cell_index, "identifier_value_with_context", location,
                    redact(match.group(0)),
                )
            )
    return found


def _output_text(output: dict) -> str:
    """Flatten one nbformat output into searchable text."""
    parts: list[str] = []
    for key in ("text", "name", "ename", "evalue"):
        value = output.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
    for bundle_key in ("data", "metadata"):
        bundle = output.get(bundle_key)
        if isinstance(bundle, dict):
            for mime, value in bundle.items():
                # Skip binary payloads; base64 images carry no readable id, and
                # decoding them here would be slow and pointless.
                if mime.startswith("image/") or mime == "application/pdf":
                    continue
                if isinstance(value, list):
                    parts.extend(str(item) for item in value)
                else:
                    parts.append(str(value))
    for item in output.get("traceback", []) or []:
        parts.append(str(item))
    return "\n".join(parts)


def check_notebook(path: Path) -> list[Violation]:
    """Return every violation in one notebook. Never mutates the file."""
    name = str(path)
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [Violation(name, -1, "unreadable_notebook", "file", type(exc).__name__)]

    violations: list[Violation] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if not isinstance(cell, dict):
            continue

        outputs = cell.get("outputs") or []
        if outputs:
            violations.append(
                Violation(name, index, "executed_output_present", "outputs", f"{len(outputs)} output(s)")
            )
        if cell.get("execution_count") is not None:
            violations.append(
                Violation(name, index, "execution_count_set", "execution_count",
                          redact(str(cell["execution_count"])))
            )

        for output in outputs:
            if isinstance(output, dict):
                violations.extend(_scan_text(_output_text(output), name, index, "outputs"))

        source = cell.get("source", "")
        source_text = "".join(source) if isinstance(source, list) else str(source)
        # Source is scanned for secrets and data paths only. Identifier *keys*
        # in source are ordinary column references (`df["subject_id"]`) and are
        # not a leak; flagging them would make the checker unusable.
        for kind, pattern in CREDENTIAL_PATTERNS:
            for match in pattern.finditer(source_text):
                violations.append(
                    Violation(name, index, kind, "source", redact(match.group(0)))
                )
        for match in MIMIC_PATH.finditer(source_text):
            violations.append(
                Violation(name, index, "mimic_data_path", "source", redact(match.group(0)))
            )
    return violations


def sanitize_notebook(source: Path, destination: Path) -> int:
    """Write a copy with outputs stripped. Cell source is left untouched.

    Returns the number of cells changed. Never edits ``source`` in place.
    """
    notebook = json.loads(source.read_text(encoding="utf-8"))
    changed = 0
    for cell in notebook.get("cells", []):
        if not isinstance(cell, dict):
            continue
        touched = False
        if cell.get("outputs"):
            cell["outputs"] = []
            touched = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            touched = True
        changed += int(touched)
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def _git(*args: str) -> list[str]:
    proc = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def staged_notebooks() -> list[Path]:
    names = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    return [Path(n) for n in names if n.endswith(".ipynb") and Path(n).is_file()]


def tracked_notebooks() -> list[Path]:
    names = _git("ls-files", "*.ipynb")
    return [Path(n) for n in names if Path(n).is_file()]


def is_tracked(path: Path) -> bool:
    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        capture_output=True, text=True, check=False,
    )
    return proc.returncode == 0


def report(violations: list[Violation]) -> int:
    if not violations:
        return EXIT_OK
    by_file: dict[str, list[Violation]] = {}
    for violation in violations:
        by_file.setdefault(violation.path, []).append(violation)

    print("NOTEBOOK PRIVACY CHECK FAILED", file=sys.stderr)
    print("", file=sys.stderr)
    for path, items in sorted(by_file.items()):
        print(f"  {path}", file=sys.stderr)
        for item in items:
            print(f"    cell {item.cell_index:>3} [{item.location}] "
                  f"{item.kind}: {item.redacted}", file=sys.stderr)
        print("", file=sys.stderr)
    print(
        "Values above are redacted on purpose. MIMIC-CXR is PhysioNet\n"
        "credentialed data and its DUA forbids redistribution; committing an\n"
        "executed notebook publishes patient identifiers and report text.\n"
        "\n"
        "To commit the code without the data:\n"
        "  python scripts/check_notebook_privacy.py --sanitize <nb> --output <clean.ipynb>\n"
        "then review the clean copy and stage that instead.",
        file=sys.stderr,
    )
    return EXIT_VIOLATION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--staged", action="store_true", help="check staged notebooks (pre-commit)")
    group.add_argument("--path", type=Path, action="append", help="check a specific notebook")
    group.add_argument("--all-tracked", action="store_true", help="check every tracked notebook")
    group.add_argument("--sanitize", type=Path, help="write an output-stripped copy")
    parser.add_argument("--output", type=Path, help="destination for --sanitize")
    parser.add_argument(
        "--overwrite-in-place",
        action="store_true",
        help="allow --sanitize to overwrite its input (refused for tracked files)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.sanitize is not None:
        source = args.sanitize
        if not source.is_file():
            print(f"no such notebook: {source}", file=sys.stderr)
            return EXIT_ERROR
        destination = args.output
        if destination is None:
            if not args.overwrite_in_place:
                print(
                    "--sanitize needs --output, or --overwrite-in-place to edit the "
                    "original. Refusing to modify a notebook implicitly.",
                    file=sys.stderr,
                )
                return EXIT_ERROR
            destination = source
        if destination == source and is_tracked(source):
            print(
                f"refusing to overwrite {source} in place: it is tracked by Git. "
                "Write to a new path with --output and review the result first.",
                file=sys.stderr,
            )
            return EXIT_ERROR
        changed = sanitize_notebook(source, destination)
        print(f"stripped outputs from {changed} cell(s) -> {destination}")
        return EXIT_OK

    if args.staged:
        targets = staged_notebooks()
    elif args.all_tracked:
        targets = tracked_notebooks()
    else:
        targets = list(args.path or [])
        missing = [t for t in targets if not t.is_file()]
        if missing:
            print(f"no such notebook(s): {', '.join(str(m) for m in missing)}", file=sys.stderr)
            return EXIT_ERROR

    violations: list[Violation] = []
    for target in targets:
        violations.extend(check_notebook(target))
    return report(violations)


if __name__ == "__main__":
    sys.exit(main())
