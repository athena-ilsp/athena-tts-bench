#!/usr/bin/env python3
"""Validate GreekTTS-Bench dataset JSON files."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from greekttsbench.datasets import dataset_files, load_json, validate_dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", default="datasets")
    parser.add_argument("--strict-word-count", action="store_true")
    parser.add_argument("--word-count-tolerance", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.datasets_dir)
    paths = dataset_files(dataset_dir)
    issues = validate_dataset_dir(
        dataset_dir,
        word_count_tolerance=args.word_count_tolerance,
        strict_word_count=args.strict_word_count,
    )

    total_sentences = 0
    scenarios: Counter[str] = Counter()
    for path in paths:
        data = load_json(path)
        total_sentences += len(data.get("sentences", []))
        scenarios[str(data.get("scenario"))] += 1

    print(f"dataset_files={len(paths)}")
    print(f"total_sentences={total_sentences}")
    print(f"scenarios={len(scenarios)}")

    counts = Counter(issue.severity for issue in issues)
    print(f"errors={counts['error']}")
    print(f"warnings={counts['warning']}")

    for issue in issues:
        location = issue.path
        if issue.sentence_id:
            location += f":{issue.sentence_id}"
        print(f"{issue.severity.upper()} {issue.code} {location} - {issue.message}")

    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
