#!/usr/bin/env python3
"""Build a JSONL manifest for TTS synthesis jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from greekttsbench.manifests import (
    build_synthesis_manifest_rows,
    enabled_systems,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", default="configs/experiment.json")
    parser.add_argument("--tts-config", default=None)
    parser.add_argument("--output", default="manifests/synthesis_manifest.jsonl")
    parser.add_argument("--method", action="append", default=None)
    parser.add_argument(
        "--exclude-scenario",
        action="append",
        default=None,
        help=(
            "Drop datasets with this scenario (repeatable), e.g. "
            "english_comparison for the Greek-core run."
        ),
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--core",
        action="store_true",
        help=(
            "Middle-ground tier: read the experiment config 'core' block "
            "(7 scenarios x 20 sentences, raw/regex/llm). Mutually exclusive with --pilot."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Keep only sentences with source token count at or below this cutoff.",
    )
    parser.add_argument(
        "--selection-seed",
        default=None,
        help=(
            "Seed for pilot sentence sampling. Defaults to the experiment "
            "config pilot.selection_seed; pass 'none' for file-order selection."
        ),
    )
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    experiment = load_config(args.experiment_config)
    tts_config = load_config(args.tts_config or experiment["tts_systems_config"])

    if args.pilot and args.core:
        raise SystemExit("--pilot and --core are mutually exclusive")

    methods = args.method or experiment["preprocessing_methods"]
    scenarios = None
    sentences_per_scenario = None
    selection_seed = None
    if args.pilot or args.core:
        tier = experiment["core" if args.core else "pilot"]
        scenarios = set(tier["scenarios"])
        sentences_per_scenario = int(tier["sentences_per_scenario"])
        methods = args.method or tier["preprocessing_methods"]
        selection_seed = tier.get("selection_seed")
    if args.selection_seed is not None:
        selection_seed = (
            None if args.selection_seed.lower() == "none" else args.selection_seed
        )

    rows, warnings = build_synthesis_manifest_rows(
        dataset_dir=experiment["dataset_dir"],
        preprocessed_dir=experiment["preprocessed_dir"],
        audio_dir=experiment["audio_dir"],
        preprocessing_methods=methods,
        systems=enabled_systems(tts_config),
        scenarios=scenarios,
        exclude_scenarios=(
            set(args.exclude_scenario) if args.exclude_scenario else None
        ),
        sentences_per_scenario=sentences_per_scenario,
        max_tokens=args.max_tokens,
        selection_seed=selection_seed,
    )

    for warning in warnings:
        print(f"WARNING {warning}")

    if warnings and args.fail_on_missing:
        return 1

    write_jsonl(args.output, rows)
    print(f"wrote {args.output}")
    print(f"rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
