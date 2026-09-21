#!/usr/bin/env python3
"""Run a synthetic normalization/scoring example without models or network."""
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from greekttsbench.datasets import load_json, validate_dataset_file
from greekttsbench.metrics import error_rates
from greekttsbench.normalization import normalize_dataset_object


def main() -> int:
    path = PROJECT_ROOT / "examples/offline/datasets/example.json"
    errors = [issue for issue in validate_dataset_file(path, strict_word_count=True)
              if issue.severity == "error"]
    if errors:
        raise SystemExit(str(errors))
    data = load_json(path)
    raw = normalize_dataset_object(data, "raw")
    regex = normalize_dataset_object(data, "regex")
    if any(row["text"] != normalized["normalized_text"]
           for row, normalized in zip(data["sentences"], raw["sentences"])):
        raise SystemExit("Raw normalization changed the input text")
    reference = data["sentences"][1]["text"]
    hypothesis = "Το δύο χιλιάδες είκοσι τέσσερα ήταν σπουδαίο έτος."
    if regex["sentences"][1]["normalized_text"] != hypothesis:
        raise SystemExit("Rule normalization differs from the expected example")
    scores = error_rates(reference, hypothesis, "el")
    if scores["wer"] != 0.0 or scores["cer"] != 0.0 or scores["wer_orthographic"] <= 0:
        raise SystemExit("Unexpected scoring-policy output")
    print(json.dumps({"reference": reference, "hypothesis": hypothesis, **scores},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
