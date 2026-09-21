import json
from pathlib import Path

import pytest

from greekttsbench.manifests import (
    build_synthesis_manifest_rows,
    build_transcription_manifest_rows,
    select_sentence_ids,
)


SYSTEMS = [
    {"id": "sys_a", "family": "fam", "display_name": "System A"},
    {"id": "sys_b", "family": "fam", "display_name": "System B"},
]

ASR_CONFIG = {
    "id": "whisperx_test",
    "backend": "whisperx",
    "model": "large-v3",
    "language": "el",
}


def make_dataset(
    tmp_path: Path,
    language: str = "el",
    n: int = 20,
    scenario: str = "test_scenario",
    file_name: str = "dataset_99_test.json",
) -> Path:
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir(exist_ok=True)
    sentences = [
        {
            "id": f"s_{i:03d}",
            "text": f"Πρόταση αριθμός {i} για τον έλεγχο.",
            "source_url": "https://example.org",
            "word_count": 6,
            "characteristics": {},
        }
        for i in range(1, n + 1)
    ]
    data = {
        "dataset_id": f"{scenario}_v1",
        "scenario": scenario,
        "language": language,
        "total_sentences": n,
        "source": ["example.org"],
        "sentences": sentences,
    }
    path = dataset_dir / file_name
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    preprocessed = tmp_path / "preprocessed"
    for method in ("raw", "regex"):
        variant = json.loads(json.dumps(data))
        for sentence in variant["sentences"]:
            sentence["original_text"] = sentence["text"]
            sentence["normalized_text"] = (
                sentence["text"] if method == "raw" else sentence["text"] + " regex"
            )
        method_dir = preprocessed / method
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / path.name).write_text(
            json.dumps(variant, ensure_ascii=False), encoding="utf-8"
        )
    return tmp_path


def build_rows(tmp_path: Path, **kwargs):
    defaults = dict(
        dataset_dir=tmp_path / "datasets",
        preprocessed_dir=tmp_path / "preprocessed",
        audio_dir=tmp_path / "audio",
        preprocessing_methods=["raw", "regex"],
        systems=SYSTEMS,
    )
    defaults.update(kwargs)
    return build_synthesis_manifest_rows(**defaults)


def test_pilot_includes_every_method_with_identical_sentences(tmp_path):
    make_dataset(tmp_path)
    rows, warnings = build_rows(
        tmp_path, sentences_per_scenario=5, selection_seed=7
    )
    assert warnings == []

    by_method = {}
    for row in rows:
        by_method.setdefault(row["preprocessing_method"], set()).add(
            row["sentence_id"]
        )
    assert set(by_method) == {"raw", "regex"}
    assert by_method["raw"] == by_method["regex"]
    assert len(by_method["raw"]) == 5
    # 5 sentences x 2 methods x 2 systems
    assert len(rows) == 20


def test_pilot_selection_is_seeded_and_deterministic(tmp_path):
    make_dataset(tmp_path)
    rows_a, _ = build_rows(tmp_path, sentences_per_scenario=5, selection_seed=7)
    rows_b, _ = build_rows(tmp_path, sentences_per_scenario=5, selection_seed=7)
    rows_c, _ = build_rows(tmp_path, sentences_per_scenario=5, selection_seed=8)

    ids_a = sorted({r["sentence_id"] for r in rows_a})
    ids_b = sorted({r["sentence_id"] for r in rows_b})
    ids_c = sorted({r["sentence_id"] for r in rows_c})
    assert ids_a == ids_b
    assert ids_a != ids_c
    # Seeded sampling should not just take the first N in file order.
    assert ids_a != [f"s_{i:03d}" for i in range(1, 6)]


def test_full_run_keeps_all_sentences_for_all_methods(tmp_path):
    make_dataset(tmp_path)
    rows, _ = build_rows(tmp_path)
    # 20 sentences x 2 methods x 2 systems
    assert len(rows) == 80


def test_exclude_scenarios_drops_matching_datasets(tmp_path):
    make_dataset(tmp_path)
    make_dataset(
        tmp_path,
        language="en",
        scenario="english_comparison",
        file_name="dataset_98_english.json",
    )

    rows, warnings = build_rows(
        tmp_path, exclude_scenarios={"english_comparison"}
    )
    assert warnings == []
    assert {row["scenario"] for row in rows} == {"test_scenario"}
    # 20 sentences x 2 methods x 2 systems, English dataset dropped entirely.
    assert len(rows) == 80


def test_select_sentence_ids_respects_max_tokens():
    sentences = [
        {"id": "short", "text": "Λίγες λέξεις εδώ."},
        {"id": "long", "text": " ".join(["λέξη"] * 40)},
    ]
    chosen = select_sentence_ids(
        sentences, limit=None, max_tokens=25, seed=None, dataset_id="d"
    )
    assert chosen == ["short"]


def test_transcription_rows_use_row_language(tmp_path):
    make_dataset(tmp_path)
    synthesis_rows, _ = build_rows(tmp_path, sentences_per_scenario=2, selection_seed=1)
    for row in synthesis_rows:
        row["language"] = "en"

    rows = build_transcription_manifest_rows(
        synthesis_rows=synthesis_rows,
        asr_config=ASR_CONFIG,
        transcript_dir=tmp_path / "transcripts",
    )
    for row in rows:
        assert row["language"] == "en"
        command = row["command"]
        assert command[command.index("--language") + 1] == "en"


def test_transcription_rows_fall_back_to_asr_language(tmp_path):
    make_dataset(tmp_path)
    synthesis_rows, _ = build_rows(tmp_path, sentences_per_scenario=2, selection_seed=1)
    for row in synthesis_rows:
        row["language"] = None

    rows = build_transcription_manifest_rows(
        synthesis_rows=synthesis_rows,
        asr_config=ASR_CONFIG,
        transcript_dir=tmp_path / "transcripts",
    )
    assert all(r["language"] == "el" for r in rows)


def test_transcription_rows_wire_align_and_diarize_flags(tmp_path):
    make_dataset(tmp_path)
    synthesis_rows, _ = build_rows(tmp_path, sentences_per_scenario=1, selection_seed=1)

    no_align = build_transcription_manifest_rows(
        synthesis_rows=synthesis_rows,
        asr_config={**ASR_CONFIG, "align": False, "diarize": True},
        transcript_dir=tmp_path / "transcripts",
    )
    assert "--no_align" in no_align[0]["command"]
    assert "--diarize" in no_align[0]["command"]

    custom_align = build_transcription_manifest_rows(
        synthesis_rows=synthesis_rows,
        asr_config={**ASR_CONFIG, "align": True, "alignment_model": "some/model"},
        transcript_dir=tmp_path / "transcripts",
    )
    command = custom_align[0]["command"]
    assert command[command.index("--align_model") + 1] == "some/model"
