"""Manifest builders for synthesis and ASR evaluation jobs."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from greekttsbench.datasets import dataset_files, iter_sentences, load_json, rough_word_count


ID_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_id(value: str) -> str:
    """Make a string safe for manifest IDs and paths."""
    return ID_SAFE_RE.sub("_", value).strip("_")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON Lines records."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write JSON Lines records."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one JSON Lines record, flushed immediately.

    Used by long-running scoring jobs so a walltime kill loses at most the
    clip currently being processed.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def enabled_systems(tts_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return enabled TTS systems from the config."""
    return [
        system
        for system in tts_config.get("systems", [])
        if system.get("enabled", True)
    ]


def dataset_variant_path(
    dataset_dir: str | Path,
    preprocessed_dir: str | Path,
    method: str,
    dataset_file: Path,
) -> Path:
    """Resolve the dataset file to use for a preprocessing method."""
    if method == "raw":
        generated_raw = Path(preprocessed_dir) / "raw" / dataset_file.name
        if generated_raw.exists():
            return generated_raw
        return Path(dataset_dir) / dataset_file.name
    return Path(preprocessed_dir) / method / dataset_file.name


def sentence_for_manifest(sentence: dict[str, Any]) -> tuple[str, str]:
    """Return source text and synthesis text for a sentence record."""
    source_text = sentence.get("original_text") or sentence["text"]
    synthesis_text = sentence.get("normalized_text") or sentence["text"]
    return source_text, synthesis_text


def select_sentence_ids(
    sentences: list[dict[str, Any]],
    *,
    limit: int | None,
    max_tokens: int | None,
    seed: int | str | None,
    dataset_id: str,
) -> list[str]:
    """Pick the sentence ids for a run, identically for every preprocessing method.

    Selection is computed once from the base dataset (source text), so every
    preprocessing condition synthesizes exactly the same sentences. When a limit
    is set (pilot runs) and a seed is given, sentences are sampled with a seeded
    shuffle instead of file order; dataset files are ordered by sub-category, so
    first-N would cover only the first sub-category. The returned ids keep file
    order for stable manifests.
    """
    eligible = [
        str(sentence["id"])
        for sentence in sentences
        if max_tokens is None
        or rough_word_count(str(sentence.get("text", ""))) <= max_tokens
    ]
    if limit is None or len(eligible) <= limit:
        return eligible

    if seed is None:
        return eligible[:limit]

    rng = random.Random(f"{seed}:{dataset_id}")
    shuffled = list(eligible)
    rng.shuffle(shuffled)
    chosen = set(shuffled[:limit])
    return [sentence_id for sentence_id in eligible if sentence_id in chosen]


def build_synthesis_manifest_rows(
    *,
    dataset_dir: str | Path,
    preprocessed_dir: str | Path,
    audio_dir: str | Path,
    preprocessing_methods: list[str],
    systems: list[dict[str, Any]],
    scenarios: set[str] | None = None,
    exclude_scenarios: set[str] | None = None,
    sentences_per_scenario: int | None = None,
    max_tokens: int | None = None,
    selection_seed: int | str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build synthesis manifest rows for the requested matrix.

    Sentence selection happens once per dataset, from the base dataset file, so
    every preprocessing method covers exactly the same sentences.
    """
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    scenario_counts: Counter[str] = Counter()

    for dataset_file in dataset_files(dataset_dir):
        base_data = load_json(dataset_file)
        scenario = str(base_data.get("scenario"))
        if scenarios is not None and scenario not in scenarios:
            continue
        if exclude_scenarios is not None and scenario in exclude_scenarios:
            continue

        # A scenario may span multiple dataset files; cap across all of them.
        limit = None
        if sentences_per_scenario is not None:
            limit = sentences_per_scenario - scenario_counts[scenario]
            if limit <= 0:
                continue

        selected_ids = select_sentence_ids(
            list(iter_sentences(base_data)),
            limit=limit,
            max_tokens=max_tokens,
            seed=selection_seed,
            dataset_id=str(base_data.get("dataset_id")),
        )
        if sentences_per_scenario is not None:
            scenario_counts[scenario] += len(selected_ids)
        if not selected_ids:
            continue

        for method in preprocessing_methods:
            variant_path = dataset_variant_path(
                dataset_dir, preprocessed_dir, method, dataset_file
            )
            if not variant_path.exists():
                warnings.append(f"missing preprocessed dataset: {variant_path}")
                continue

            data = load_json(variant_path)
            dataset_id = str(data["dataset_id"])
            sentences_by_id = {
                str(sentence["id"]): sentence for sentence in iter_sentences(data)
            }

            for sentence_id in selected_ids:
                sentence = sentences_by_id.get(sentence_id)
                if sentence is None:
                    warnings.append(
                        f"sentence {sentence_id} missing in {variant_path}"
                    )
                    continue

                source_text, synthesis_text = sentence_for_manifest(sentence)
                source_token_count = rough_word_count(source_text)
                synthesis_token_count = rough_word_count(synthesis_text)

                for system in systems:
                    system_id = str(system["id"])
                    utterance_id = safe_id(
                        f"{dataset_id}_{sentence_id}_{method}_{system_id}"
                    )
                    audio_path = (
                        Path(audio_dir)
                        / system_id
                        / method
                        / dataset_id
                        / f"{sentence_id}.wav"
                    )
                    rows.append(
                        {
                            "utterance_id": utterance_id,
                            "dataset_id": dataset_id,
                            "dataset_file": dataset_file.name,
                            "scenario": scenario,
                            "sentence_id": sentence_id,
                            "language": data.get("language"),
                            "preprocessing_method": method,
                            "tts_system_id": system_id,
                            "tts_family": system.get("family"),
                            "tts_display_name": system.get("display_name"),
                            "source_text": source_text,
                            "synthesis_text": synthesis_text,
                            "source_token_count": source_token_count,
                            "synthesis_token_count": synthesis_token_count,
                            "source_url": sentence.get("source_url"),
                            "audio_path": str(audio_path),
                            "metadata": {
                                "model_ref": system.get("model_ref"),
                                "voice": system.get("voice"),
                                "sample_rate_hz": system.get("sample_rate_hz"),
                            },
                        }
                    )

    return rows, warnings


def build_transcription_manifest_rows(
    *,
    synthesis_rows: list[dict[str, Any]],
    asr_config: dict[str, Any],
    transcript_dir: str | Path,
) -> list[dict[str, Any]]:
    """Build WhisperX transcription manifest rows from synthesis rows."""
    rows: list[dict[str, Any]] = []
    asr_id = str(asr_config["id"])

    for row in synthesis_rows:
        transcript_path = (
            Path(transcript_dir)
            / asr_id
            / row["tts_system_id"]
            / row["preprocessing_method"]
            / row["dataset_id"]
            / f"{row['sentence_id']}.json"
        )
        output_dir = transcript_path.parent
        # Transcribe each clip in its own language (the English comparison arm
        # must not be force-decoded as Greek); fall back to the ASR default.
        language = str(row.get("language") or asr_config["language"])
        command = [
            "whisperx",
            row["audio_path"],
            "--model",
            str(asr_config["model"]),
            "--language",
            language,
            "--task",
            str(asr_config.get("task", "transcribe")),
            "--output_format",
            str(asr_config.get("output_format", "json")),
            "--output_dir",
            str(output_dir),
            "--device",
            str(asr_config.get("device", "cuda")),
            "--compute_type",
            str(asr_config.get("compute_type", "float16")),
            "--batch_size",
            str(asr_config.get("batch_size", 16)),
        ]
        if not asr_config.get("align", True):
            command.append("--no_align")
        elif asr_config.get("alignment_model"):
            command.extend(["--align_model", str(asr_config["alignment_model"])])
        if asr_config.get("diarize"):
            command.append("--diarize")

        rows.append(
            {
                "utterance_id": row["utterance_id"],
                "asr_id": asr_id,
                "asr_backend": asr_config.get("backend"),
                "asr_model": asr_config.get("model"),
                "language": language,
                "audio_path": row["audio_path"],
                "transcript_path": str(transcript_path),
                "expected_text": row["synthesis_text"],
                "source_text": row["source_text"],
                "dataset_id": row["dataset_id"],
                "scenario": row["scenario"],
                "sentence_id": row["sentence_id"],
                "preprocessing_method": row["preprocessing_method"],
                "tts_system_id": row["tts_system_id"],
                "command": command,
            }
        )

    return rows
