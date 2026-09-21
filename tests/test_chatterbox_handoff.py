import json
from pathlib import Path

from greekttsbench.chatterbox_handoff import (
    copy_outputs,
    export_handoff,
    summarize_audio_presence,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def manifest_row(system_id: str, sentence_id: str, audio_path: Path) -> dict:
    return {
        "utterance_id": f"{sentence_id}_{system_id}",
        "tts_system_id": system_id,
        "tts_family": "chatterbox" if system_id.startswith("chatterbox") else "vits",
        "sentence_id": sentence_id,
        "preprocessing_method": "raw",
        "scenario": "numbers_dates",
        "synthesis_text": "Hello   from\nChatterbox",
        "audio_path": str(audio_path),
    }


def test_export_handoff_writes_texts_and_audio_maps(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        manifest_row("chatterbox_el_a", "s001", tmp_path / "a" / "s001.wav"),
        manifest_row("vits_el_1gpu", "s001", tmp_path / "vits" / "s001.wav"),
        manifest_row("chatterbox_el_a", "s002", tmp_path / "a" / "s002.wav"),
        manifest_row("chatterbox_el_b", "s001", tmp_path / "b" / "s001.wav"),
    ]
    write_jsonl(manifest, rows)

    summaries = export_handoff(
        manifest_path=manifest,
        system_ids=("chatterbox_el_a", "chatterbox_el_b"),
        out_dir=tmp_path / "handoff",
    )

    assert [summary.rows for summary in summaries] == [2, 1]
    assert (tmp_path / "handoff" / "chatterbox_el_a.txt").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "Hello from Chatterbox",
        "Hello from Chatterbox",
    ]

    audio_map = [
        json.loads(line)
        for line in (tmp_path / "handoff" / "chatterbox_el_a_audio_map.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["index"] for row in audio_map] == [1, 2]
    assert [row["sentence_id"] for row in audio_map] == ["s001", "s002"]


def test_copy_outputs_uses_variant_filenames(tmp_path):
    audio_map = tmp_path / "handoff" / "chatterbox_el_a_audio_map.jsonl"
    destination = tmp_path / "artifacts" / "audio" / "chatterbox_el_a" / "s001.wav"
    write_jsonl(
        audio_map,
        [
            {
                "index": 1,
                "audio_path": str(destination),
            }
        ],
    )
    source = tmp_path / "run" / "male" / "001_ft.wav"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fake wav")

    summary = copy_outputs(
        system_id="chatterbox_el_a",
        audio_map_path=audio_map,
        run_dir=tmp_path / "run",
        speaker="male",
        variant="ft",
    )

    assert summary.copied == 1
    assert summary.missing == ()
    assert destination.read_bytes() == b"fake wav"


def test_summarize_audio_presence_counts_by_system_and_family(tmp_path):
    existing = tmp_path / "audio" / "chatterbox_el_a" / "s001.wav"
    missing = tmp_path / "audio" / "chatterbox_el_b" / "s001.wav"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"fake wav")

    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [
            manifest_row("chatterbox_el_a", "s001", existing),
            manifest_row("chatterbox_el_b", "s001", missing),
        ],
    )

    summary = summarize_audio_presence(manifest_path=manifest)

    assert summary["total"] == {"rows": 2, "existing": 1, "missing": 1}
    assert summary["by_system"]["chatterbox_el_a"]["existing"] == 1
    assert summary["by_system"]["chatterbox_el_b"]["missing"] == 1
    assert summary["by_family"]["chatterbox"] == {
        "rows": 2,
        "existing": 1,
        "missing": 1,
    }
