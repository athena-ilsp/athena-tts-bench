#!/usr/bin/env python3
"""Score WhisperX back-transcriptions against synthesis text.

Reports two metric families per clip:

- wer/cer: scoring-policy metrics. Digits, dates, times, percentages,
  currencies, and abbreviations are verbalized identically in reference and
  hypothesis (Greek), and accents are folded, so preprocessing conditions are
  comparable even though their references differ orthographically.
- wer_orthographic/cer_orthographic: plain case/punctuation normalization
  only, documenting how much the policy normalization mattered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from greekttsbench.manifests import load_jsonl, write_jsonl
from greekttsbench.metrics import (
    error_rates,
    extract_whisperx_text,
    summarize_error_rates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/transcription_manifest.jsonl")
    parser.add_argument("--output", default="results/asr_scores.jsonl")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--exclude-empty-hypothesis",
        action="store_true",
        help=(
            "Skip transcripts whose extracted hypothesis text is empty. Useful "
            "when no-speech/VAD failures were written as empty transcript JSONs."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.manifest)
    scored = []
    missing = []
    excluded_empty = []

    for row in rows:
        transcript_path = Path(row["transcript_path"])
        if not transcript_path.exists():
            missing.append(str(transcript_path))
            continue

        hypothesis = extract_whisperx_text(transcript_path)
        if args.exclude_empty_hypothesis and not hypothesis.strip():
            excluded_empty.append(str(transcript_path))
            continue

        reference = row["expected_text"]
        language = str(row.get("language") or "el")
        scored.append(
            {
                **{
                    key: row[key]
                    for key in (
                        "utterance_id",
                        "dataset_id",
                        "scenario",
                        "sentence_id",
                        "preprocessing_method",
                        "tts_system_id",
                        "asr_id",
                    )
                },
                "language": language,
                "reference": reference,
                "hypothesis": hypothesis,
                **error_rates(reference, hypothesis, language),
            }
        )

    write_jsonl(args.output, scored)
    print(f"wrote {args.output}")
    print(f"scored={len(scored)}")
    print(f"missing={len(missing)}")
    print(f"excluded_empty_hypothesis={len(excluded_empty)}")
    if missing and not args.allow_missing:
        print("First missing transcripts:")
        for path in missing[:20]:
            print(path)
        return 1

    summary = summarize_error_rates(scored)
    summary_path = Path(args.output).with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scored": len(scored),
                "missing": len(missing),
                "excluded_empty_hypothesis": len(excluded_empty),
                **summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")
    print(f"wrote {summary_path}")
    overall = summary["overall"]
    print(
        f"mean_wer={overall['mean_wer']} "
        f"mean_wer_orthographic={overall['mean_wer_orthographic']} "
        f"over {overall['count']} clips"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
