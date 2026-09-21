"""Dataset loading and validation utilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REQUIRED_TOP_LEVEL_FIELDS = {
    "dataset_id",
    "scenario",
    "language",
    "total_sentences",
    "source",
    "sentences",
}

REQUIRED_SENTENCE_FIELDS = {
    "id",
    "text",
    "source_url",
    "word_count",
    "characteristics",
}

WORD_RE = re.compile(r"[\w]+(?:[./-][\w]+)*", re.UNICODE)


@dataclass(frozen=True)
class ValidationIssue:
    """A validation finding for a dataset file."""

    path: str
    severity: str
    code: str
    message: str
    sentence_id: str | None = None


def dataset_files(dataset_dir: str | Path) -> list[Path]:
    """Return benchmark JSON dataset files in stable order."""
    return sorted(Path(dataset_dir).glob("*.json"))


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write a UTF-8 JSON object with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def rough_word_count(text: str) -> int:
    """Count whitespace-independent word-like tokens for metadata checks."""
    return len(WORD_RE.findall(text))


def iter_sentences(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield sentence records from a dataset object."""
    sentences = data.get("sentences", [])
    if not isinstance(sentences, list):
        return
    for sentence in sentences:
        if isinstance(sentence, dict):
            yield sentence


def validate_dataset_file(
    path: str | Path,
    *,
    word_count_tolerance: int = 2,
    strict_word_count: bool = False,
) -> list[ValidationIssue]:
    """Validate one dataset JSON file.

    Word counts are warnings by default because legacy metadata may have been
    produced by a different tokenizer. Use strict_word_count=True before release.
    """
    path = Path(path)
    issues: list[ValidationIssue] = []

    try:
        data = load_json(path)
    except Exception as exc:
        return [
            ValidationIssue(
                path=str(path),
                severity="error",
                code="json_error",
                message=str(exc),
            )
        ]

    missing_top = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(data))
    for field in missing_top:
        issues.append(
            ValidationIssue(
                path=str(path),
                severity="error",
                code="missing_top_level_field",
                message=f"missing top-level field: {field}",
            )
        )

    sentences = data.get("sentences")
    if not isinstance(sentences, list):
        issues.append(
            ValidationIssue(
                path=str(path),
                severity="error",
                code="sentences_not_list",
                message="'sentences' must be a list",
            )
        )
        return issues

    declared_total = data.get("total_sentences")
    if declared_total != len(sentences):
        issues.append(
            ValidationIssue(
                path=str(path),
                severity="error",
                code="sentence_count_mismatch",
                message=f"total_sentences={declared_total}, actual={len(sentences)}",
            )
        )

    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()

    for index, sentence in enumerate(sentences):
        if not isinstance(sentence, dict):
            issues.append(
                ValidationIssue(
                    path=str(path),
                    severity="error",
                    code="sentence_not_object",
                    message=f"sentence at index {index} is not an object",
                )
            )
            continue

        sentence_id = str(sentence.get("id", f"index_{index}"))
        if sentence_id in seen_ids:
            duplicate_ids.add(sentence_id)
        seen_ids.add(sentence_id)

        for field in sorted(REQUIRED_SENTENCE_FIELDS - set(sentence)):
            issues.append(
                ValidationIssue(
                    path=str(path),
                    severity="error",
                    code="missing_sentence_field",
                    message=f"missing sentence field: {field}",
                    sentence_id=sentence_id,
                )
            )

        if not isinstance(sentence.get("text"), str) or not sentence.get("text"):
            issues.append(
                ValidationIssue(
                    path=str(path),
                    severity="error",
                    code="empty_text",
                    message="sentence text must be a non-empty string",
                    sentence_id=sentence_id,
                )
            )

        if "word_count" in sentence and isinstance(sentence.get("text"), str):
            actual_word_count = rough_word_count(sentence["text"])
            recorded_word_count = sentence["word_count"]
            if not isinstance(recorded_word_count, int):
                issues.append(
                    ValidationIssue(
                        path=str(path),
                        severity="error",
                        code="word_count_not_int",
                        message="word_count must be an integer",
                        sentence_id=sentence_id,
                    )
                )
            elif abs(recorded_word_count - actual_word_count) > word_count_tolerance:
                issues.append(
                    ValidationIssue(
                        path=str(path),
                        severity="error" if strict_word_count else "warning",
                        code="word_count_mismatch",
                        message=(
                            f"word_count={recorded_word_count}, "
                            f"rough_count={actual_word_count}"
                        ),
                        sentence_id=sentence_id,
                    )
                )

        if "characteristics" in sentence and not isinstance(sentence["characteristics"], dict):
            issues.append(
                ValidationIssue(
                    path=str(path),
                    severity="error",
                    code="characteristics_not_object",
                    message="characteristics must be an object",
                    sentence_id=sentence_id,
                )
            )

    for sentence_id in sorted(duplicate_ids):
        issues.append(
            ValidationIssue(
                path=str(path),
                severity="error",
                code="duplicate_sentence_id",
                message=f"duplicate sentence id: {sentence_id}",
                sentence_id=sentence_id,
            )
        )

    return issues


def validate_dataset_dir(
    dataset_dir: str | Path,
    *,
    word_count_tolerance: int = 2,
    strict_word_count: bool = False,
) -> list[ValidationIssue]:
    """Validate all dataset JSON files in a directory."""
    issues: list[ValidationIssue] = []
    for path in dataset_files(dataset_dir):
        issues.extend(
            validate_dataset_file(
                path,
                word_count_tolerance=word_count_tolerance,
                strict_word_count=strict_word_count,
            )
        )
    return issues

