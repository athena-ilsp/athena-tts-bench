"""Text normalization and edit-distance metrics for ASR back-transcription."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

GREEK_LANGUAGE_CODES = {"el", "ell", "gre", "greek"}


def normalize_for_error_rate(text: str) -> str:
    """Normalize text before WER/CER scoring (orthographic: case/punctuation)."""
    text = text.lower()
    text = PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def fold_accents(text: str) -> str:
    """Remove diacritics (Greek tonos etc.) and map final sigma to sigma."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", stripped).replace("ς", "σ")


_SCORING_VERBALIZER = None


def _greek_verbalizer():
    """Shared Greek verbalizer for scoring (lazy singleton).

    Acronym expansion stays OFF: spelling out "ΟΗΕ" and saying the full name
    are genuinely different spoken forms, and equating them would mask real
    behavior differences between preprocessing conditions.
    """
    global _SCORING_VERBALIZER
    if _SCORING_VERBALIZER is None:
        from preprocessing.regex_normalizer import GreekRegexNormalizer

        _SCORING_VERBALIZER = GreekRegexNormalizer(expand_acronyms=False)
    return _SCORING_VERBALIZER


def normalize_for_scoring(text: str, language: str = "el") -> str:
    """Scoring-policy normalization applied identically to reference and hypothesis.

    The raw condition keeps digits in its references while regex/LLM references
    are verbalized, and Whisper itself often writes digits for spoken numbers.
    Comparing conditions on orthographic WER alone therefore biases against
    preprocessing. This policy verbalizes digits/dates/times/percentages/
    currencies/abbreviations on BOTH sides for Greek, folds accents (ASR accent
    variance should not count against TTS), lowercases, and strips punctuation.
    Non-Greek text gets the same treatment minus verbalization, so digits stay
    digits on both sides.
    """
    if str(language).lower() in GREEK_LANGUAGE_CODES:
        text = _greek_verbalizer().normalize(text)
    text = text.lower()
    text = fold_accents(text)
    text = PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Compute Levenshtein distance for token lists."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute WER after default text normalization."""
    ref_tokens = normalize_for_error_rate(reference).split()
    hyp_tokens = normalize_for_error_rate(hypothesis).split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute CER after default text normalization."""
    ref_chars = list(normalize_for_error_rate(reference).replace(" ", ""))
    hyp_chars = list(normalize_for_error_rate(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def _rates_for(reference: str, hypothesis: str) -> tuple[float, float]:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens:
        wer = 0.0 if not hyp_tokens else 1.0
    else:
        wer = edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    if not ref_chars:
        cer = 0.0 if not hyp_chars else 1.0
    else:
        cer = edit_distance(ref_chars, hyp_chars) / len(ref_chars)
    return wer, cer


def error_rates(
    reference: str, hypothesis: str, language: str = "el"
) -> dict[str, float]:
    """WER/CER under the scoring policy plus plain orthographic WER/CER.

    Report the policy numbers as the primary metric; the orthographic numbers
    document how much the policy normalization mattered.
    """
    policy_wer, policy_cer = _rates_for(
        normalize_for_scoring(reference, language),
        normalize_for_scoring(hypothesis, language),
    )
    ortho_wer, ortho_cer = _rates_for(
        normalize_for_error_rate(reference),
        normalize_for_error_rate(hypothesis),
    )
    return {
        "wer": policy_wer,
        "cer": policy_cer,
        "wer_orthographic": ortho_wer,
        "cer_orthographic": ortho_cer,
    }


ERROR_RATE_KEYS = ("wer", "cer", "wer_orthographic", "cer_orthographic")


def _mean_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"count": len(rows)}
    for key in ERROR_RATE_KEYS:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        aggregate[f"mean_{key}"] = (
            round(sum(values) / len(values), 4) if values else None
        )
    return aggregate


def summarize_error_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate ASR scores overall and by system, method, and scenario."""

    def group(key_fn) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[key_fn(row)].append(row)
        return {
            key: _mean_rates(group_rows)
            for key, group_rows in sorted(buckets.items())
        }

    return {
        "overall": _mean_rates(rows),
        "by_system": group(lambda r: str(r.get("tts_system_id"))),
        "by_method": group(lambda r: str(r.get("preprocessing_method"))),
        "by_system_method": group(
            lambda r: f"{r.get('tts_system_id')}::{r.get('preprocessing_method')}"
        ),
        "by_scenario": group(lambda r: str(r.get("scenario"))),
        "by_scenario_method": group(
            lambda r: f"{r.get('scenario')}::{r.get('preprocessing_method')}"
        ),
    }


def extract_whisperx_text(path: str | Path) -> str:
    """Extract transcript text from a WhisperX JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    if isinstance(data.get("text"), str):
        return data["text"]

    segments = data.get("segments", [])
    if isinstance(segments, list):
        texts = [
            segment.get("text", "")
            for segment in segments
            if isinstance(segment, dict)
        ]
        return " ".join(texts).strip()

    return ""

