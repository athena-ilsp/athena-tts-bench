#!/usr/bin/env python3
"""Build a JSONL manifest for WhisperX transcription jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from greekttsbench.manifests import (
    build_transcription_manifest_rows,
    load_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", default="configs/experiment.json")
    parser.add_argument("--asr-config", default=None)
    parser.add_argument("--synthesis-manifest", default="manifests/synthesis_manifest.jsonl")
    parser.add_argument("--output", default="manifests/transcription_manifest.jsonl")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    experiment = load_config(args.experiment_config)
    asr_config = load_config(args.asr_config or experiment["asr_config"])
    synthesis_rows = load_jsonl(args.synthesis_manifest)
    rows = build_transcription_manifest_rows(
        synthesis_rows=synthesis_rows,
        asr_config=asr_config,
        transcript_dir=experiment["transcript_dir"],
    )
    write_jsonl(args.output, rows)
    print(f"wrote {args.output}")
    print(f"rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
