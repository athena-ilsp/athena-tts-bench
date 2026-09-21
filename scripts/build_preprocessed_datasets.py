#!/usr/bin/env python3
"""Create raw, regex, or LLM-normalized dataset variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from greekttsbench.datasets import dataset_files, load_json, write_json
from greekttsbench.normalization import SUPPORTED_PREPROCESSING_METHODS, normalize_dataset_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="datasets")
    parser.add_argument("--output-dir", default="artifacts/preprocessed")
    parser.add_argument(
        "--method",
        action="append",
        choices=sorted(SUPPORTED_PREPROCESSING_METHODS),
        required=True,
        help="Preprocessing method to build. Repeat for multiple methods.",
    )
    parser.add_argument(
        "--llm-config",
        default="configs/llm_normalizer.json",
        help="JSON config for the 'llm' method (provider, model, runtime settings).",
    )
    parser.add_argument("--llm-provider", default=None, help="Override config provider.")
    parser.add_argument("--llm-model", default=None, help="Override config model / path.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_llm_preprocessor(args: argparse.Namespace):
    """Construct the LLM preprocessor once so the model loads a single time."""
    from preprocessing.llm_preprocessor import LLMPreprocessor

    config = load_config(args.llm_config)
    provider = args.llm_provider or config.get("provider", "openai")
    model = args.llm_model or config.get("model", "gpt-4o")
    print(f"LLM normalization: provider={provider} model={model}")
    return LLMPreprocessor(
        provider=provider,
        model=model,
        base_url=config.get("base_url"),
        temperature=config.get("temperature", 0.0),
        max_tokens=config.get("max_new_tokens", config.get("max_tokens", 2048)),
        device_map=config.get("device_map", "auto"),
        torch_dtype=config.get("torch_dtype", "auto"),
    )


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    llm_preprocessor = build_llm_preprocessor(args) if "llm" in args.method else None

    for method in args.method:
        method_dir = output_dir / method
        for path in dataset_files(input_dir):
            data = load_json(path)
            normalized = normalize_dataset_object(
                data,
                method,
                llm_preprocessor=llm_preprocessor,
            )
            output_path = method_dir / path.name
            write_json(output_path, normalized)
            print(f"wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
