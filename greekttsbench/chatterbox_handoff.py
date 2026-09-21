"""Helpers for moving Chatterbox renders into synthesis manifests."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from greekttsbench.manifests import load_jsonl

DEFAULT_SYNTHESIS_MANIFEST = Path(
    "manifests/synthesis_pilot_all_conditions_max25_parler4.jsonl"
)
DEFAULT_SYSTEM_IDS = ("chatterbox_el_a", "chatterbox_el_b")


@dataclass(frozen=True)
class HandoffSummary:
    """Summary of one exported Chatterbox handoff slot."""

    system_id: str
    rows: int
    texts_file: Path
    audio_map: Path


@dataclass(frozen=True)
class CopySummary:
    """Summary of one copy-back operation."""

    system_id: str
    variant: str
    run_dir: Path
    speaker: str
    copied: int
    missing: tuple[Path, ...]
    dry_run: bool = False


def normalize_synthesis_text(text: str) -> str:
    """Collapse whitespace for Chatterbox's one-text-per-line input format."""
    return " ".join(text.split())


def export_handoff(
    *,
    manifest_path: str | Path,
    system_ids: Iterable[str],
    out_dir: str | Path,
) -> list[HandoffSummary]:
    """Export Chatterbox text files and index-to-audio-path maps."""
    rows = load_jsonl(manifest_path)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[HandoffSummary] = []
    for system_id in system_ids:
        texts_file = output_dir / f"{system_id}.txt"
        audio_map = output_dir / f"{system_id}_audio_map.jsonl"
        kept = 0
        with texts_file.open("w", encoding="utf-8") as txt, audio_map.open(
            "w", encoding="utf-8"
        ) as mapping:
            for row in rows:
                if row.get("tts_system_id") != system_id:
                    continue
                kept += 1
                text = normalize_synthesis_text(str(row["synthesis_text"]))
                txt.write(text + "\n")
                mapping.write(
                    json.dumps(
                        {
                            "index": kept,
                            "utterance_id": row.get("utterance_id"),
                            "audio_path": row["audio_path"],
                            "sentence_id": row.get("sentence_id"),
                            "preprocessing_method": row.get(
                                "preprocessing_method"
                            ),
                            "scenario": row.get("scenario"),
                            "synthesis_text": text,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                mapping.write("\n")

        summaries.append(
            HandoffSummary(
                system_id=system_id,
                rows=kept,
                texts_file=texts_file,
                audio_map=audio_map,
            )
        )

    return summaries


def chatterbox_source_wav(
    *,
    run_dir: str | Path,
    speaker: str,
    index: int,
    variant: str,
) -> Path:
    """Return the expected Chatterbox eval output path for one item."""
    run_path = Path(run_dir)
    if variant == "base":
        filename = f"{index:03d}_base.wav"
    elif variant == "ft":
        filename = f"{index:03d}_ft.wav"
    elif variant == "plain":
        filename = f"{index:03d}.wav"
    else:
        raise ValueError(f"unsupported Chatterbox variant: {variant}")
    return run_path / speaker / filename


def copy_outputs(
    *,
    system_id: str,
    audio_map_path: str | Path,
    run_dir: str | Path,
    speaker: str,
    variant: str,
    dry_run: bool = False,
) -> CopySummary:
    """Copy Chatterbox WAVs from an eval run into manifest audio paths."""
    copied = 0
    missing: list[Path] = []
    rows = load_jsonl(audio_map_path)

    for row in rows:
        src = chatterbox_source_wav(
            run_dir=run_dir,
            speaker=speaker,
            index=int(row["index"]),
            variant=variant,
        )
        dst = Path(row["audio_path"])
        if not src.exists():
            missing.append(src)
            continue
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied += 1

    return CopySummary(
        system_id=system_id,
        variant=variant,
        run_dir=Path(run_dir),
        speaker=speaker,
        copied=copied,
        missing=tuple(missing),
        dry_run=dry_run,
    )


def summarize_audio_presence(
    *,
    manifest_path: str | Path,
    system_ids: Iterable[str] | None = None,
) -> dict[str, object]:
    """Count manifest rows and existing audio files, optionally by system."""
    wanted = set(system_ids) if system_ids is not None else None
    rows = load_jsonl(manifest_path)
    by_system: dict[str, Counter[str]] = {}
    by_family: dict[str, Counter[str]] = {}
    total = Counter()

    for row in rows:
        system_id = str(row.get("tts_system_id"))
        if wanted is not None and system_id not in wanted:
            continue
        family = str(row.get("tts_family") or "")
        exists = Path(row["audio_path"]).exists()

        total["rows"] += 1
        total["existing" if exists else "missing"] += 1

        system_counter = by_system.setdefault(system_id, Counter())
        system_counter["rows"] += 1
        system_counter["existing" if exists else "missing"] += 1

        family_counter = by_family.setdefault(family, Counter())
        family_counter["rows"] += 1
        family_counter["existing" if exists else "missing"] += 1

    return {
        "manifest": str(manifest_path),
        "total": dict(total),
        "by_system": {
            key: dict(counter) for key, counter in sorted(by_system.items())
        },
        "by_family": {
            key: dict(counter) for key, counter in sorted(by_family.items())
        },
    }
