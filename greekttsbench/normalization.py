"""Build raw, regex, and LLM preprocessed dataset variants."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from preprocessing.regex_normalizer import GreekRegexNormalizer


SUPPORTED_PREPROCESSING_METHODS = {"raw", "regex", "llm"}

# Normalizers below are Greek-specific. Non-Greek datasets (the English
# comparison arm) pass through unchanged so Greek expansions never leak into
# them.
GREEK_LANGUAGE_CODES = {"el", "ell", "gre", "greek"}


def normalize_text(
    text: str,
    method: str,
    *,
    regex_normalizer: GreekRegexNormalizer | None = None,
    llm_preprocessor: Any | None = None,
) -> str:
    """Normalize one text string using a named preprocessing condition."""
    if method == "raw":
        return text
    if method == "regex":
        normalizer = regex_normalizer or GreekRegexNormalizer()
        return normalizer.normalize(text)
    if method == "llm":
        if llm_preprocessor is None:
            raise ValueError("llm_preprocessor is required for method='llm'")
        return llm_preprocessor.normalize(text)
    raise ValueError(f"unsupported preprocessing method: {method}")


def normalize_dataset_object(
    data: dict[str, Any],
    method: str,
    *,
    provider: str = "openai",
    model: str = "gpt-4o",
    regex_normalizer: GreekRegexNormalizer | None = None,
    llm_preprocessor: Any | None = None,
) -> dict[str, Any]:
    """Return a dataset object with original_text and normalized_text fields."""
    if method not in SUPPORTED_PREPROCESSING_METHODS:
        raise ValueError(f"unsupported preprocessing method: {method}")

    output = deepcopy(data)
    language = str(output.get("language", "el")).lower()

    # Non-Greek datasets pass through: the regex rules and LLM prompt are
    # Greek-specific and would corrupt the English comparison arm.
    if method != "raw" and language not in GREEK_LANGUAGE_CODES:
        for sentence in output["sentences"]:
            sentence["original_text"] = sentence["text"]
            sentence["normalized_text"] = sentence["text"]
        output["preprocessing"] = {
            "method": method,
            "normalizer": "identity_non_greek",
            "note": (
                f"language={language!r} is not Greek; Greek normalization "
                "was skipped and text passed through unchanged"
            ),
        }
        return output

    regex_normalizer = regex_normalizer or GreekRegexNormalizer()

    if method == "llm" and llm_preprocessor is None:
        from preprocessing.llm_preprocessor import LLMPreprocessor

        llm_preprocessor = LLMPreprocessor(provider=provider, model=model)

    llm_warning_count = 0
    for sentence in output["sentences"]:
        sentence["original_text"] = sentence["text"]
        if method == "llm" and hasattr(llm_preprocessor, "normalize_with_report"):
            normalized, warnings = llm_preprocessor.normalize_with_report(
                sentence["text"]
            )
            sentence["normalized_text"] = normalized
            if warnings:
                sentence["normalization_warnings"] = warnings
                llm_warning_count += 1
        else:
            sentence["normalized_text"] = normalize_text(
                sentence["text"],
                method,
                regex_normalizer=regex_normalizer,
                llm_preprocessor=llm_preprocessor,
            )

    output["preprocessing"] = {
        "method": method,
        "normalizer": {
            "raw": "identity",
            "regex": "GreekRegexNormalizer",
            "llm": "LLMPreprocessor",
        }[method],
    }
    if method == "llm":
        output["preprocessing"].update(
            {
                "provider": getattr(llm_preprocessor, "provider", provider),
                "model": getattr(llm_preprocessor, "model", model),
                "sentences_with_warnings": llm_warning_count,
            }
        )

    return output

