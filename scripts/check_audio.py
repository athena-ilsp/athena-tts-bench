#!/usr/bin/env python3
"""Audio QC for synthesized WAVs listed in a synthesis manifest.

Validates the things the evaluation protocol requires before ASR/MOS runs:
files exist and parse, durations are sane (the --max-tokens filter is only an
approximate 10-second proxy), sample rates are consistent per system, and
clips are neither silent nor clipped. Uses only the standard library, so it
runs on login nodes without the ML environment.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import struct
import sys
import wave
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from greekttsbench.manifests import load_jsonl, write_jsonl


IEEE_FLOAT_WAV_FORMAT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/synthesis_manifest.jsonl")
    parser.add_argument("--output", default="results/audio_qc.jsonl")
    parser.add_argument(
        "--min-duration", type=float, default=0.3, help="Seconds; shorter clips are flagged."
    )
    parser.add_argument(
        "--max-duration", type=float, default=20.0, help="Seconds; longer clips are flagged."
    )
    parser.add_argument(
        "--silence-peak",
        type=float,
        default=0.01,
        help="Peak amplitude (0-1) below which a clip is flagged as near-silent.",
    )
    return parser.parse_args()


def read_chunk_header(handle) -> tuple[bytes, int] | None:
    """Read one RIFF chunk header, returning None at clean EOF."""
    header = handle.read(8)
    if not header:
        return None
    if len(header) != 8:
        raise EOFError("truncated chunk header")
    chunk_id, chunk_size = struct.unpack("<4sI", header)
    return chunk_id, chunk_size


def inspect_float_wav(path: Path) -> dict:
    """Inspect IEEE-float WAVs, which Python's wave module rejects."""
    record: dict = {"issues": []}
    with path.open("rb") as f:
        riff = f.read(12)
        if len(riff) != 12:
            raise EOFError("truncated RIFF header")
        riff_id, _riff_size, wave_id = struct.unpack("<4sI4s", riff)
        if riff_id != b"RIFF" or wave_id != b"WAVE":
            raise wave.Error("not a RIFF/WAVE file")

        fmt: tuple[int, int, int, int, int, int] | None = None
        data: bytes | None = None
        while True:
            header = read_chunk_header(f)
            if header is None:
                break
            chunk_id, chunk_size = header
            payload = f.read(chunk_size)
            if len(payload) != chunk_size:
                raise EOFError(f"truncated {chunk_id!r} chunk")
            if chunk_size % 2:
                f.seek(1, 1)

            if chunk_id == b"fmt ":
                if chunk_size < 16:
                    raise wave.Error("truncated fmt chunk")
                fmt = struct.unpack("<HHIIHH", payload[:16])
            elif chunk_id == b"data":
                data = payload

            if fmt is not None and data is not None:
                break

    if fmt is None:
        raise wave.Error("missing fmt chunk")
    if data is None:
        raise wave.Error("missing data chunk")

    audio_format, channels, rate, _byte_rate, block_align, bits_per_sample = fmt
    if audio_format != IEEE_FLOAT_WAV_FORMAT:
        raise wave.Error(f"unknown format: {audio_format}")
    if bits_per_sample not in (32, 64):
        raise wave.Error(f"unsupported float WAV bit depth: {bits_per_sample}")
    if channels <= 0 or rate <= 0 or block_align <= 0:
        raise wave.Error("invalid WAV metadata")

    frames = len(data) // block_align
    duration = frames / float(rate)
    width = bits_per_sample // 8
    sample_count = len(data) // width
    unpack_format = "<f" if bits_per_sample == 32 else "<d"
    peak = 0.0
    non_finite = False
    for (sample,) in struct.iter_unpack(unpack_format, data[: sample_count * width]):
        if not math.isfinite(sample):
            non_finite = True
            continue
        peak = max(peak, abs(sample))

    record.update(
        {
            "duration_s": round(duration, 3),
            "sample_rate_hz": rate,
            "channels": channels,
            "sample_width_bytes": width,
            "wav_format": "ieee_float",
            "peak_amplitude": round(peak, 4),
        }
    )
    if non_finite:
        record["issues"].append("contains non-finite samples")
    return record


def inspect_wav(path: Path, args: argparse.Namespace) -> dict:
    """Return QC measurements and issue flags for one WAV file."""
    record: dict = {"issues": []}
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            duration = frames / float(rate) if rate else 0.0
            record.update(
                {
                    "duration_s": round(duration, 3),
                    "sample_rate_hz": rate,
                    "channels": channels,
                    "sample_width_bytes": width,
                }
            )

            peak = None
            if width == 2 and frames:
                samples = array.array("h")
                samples.frombytes(wav.readframes(frames))
                if sys.byteorder == "big":
                    samples.byteswap()
                peak = max(abs(s) for s in samples) / 32768.0
                record["peak_amplitude"] = round(peak, 4)
    except wave.Error as exc:
        if "unknown format: 3" not in str(exc):
            record["issues"].append(f"unreadable: {exc}")
            return record
        try:
            record = inspect_float_wav(path)
            peak = record.get("peak_amplitude")
            duration = float(record.get("duration_s", 0.0))
        except (wave.Error, EOFError, OSError, struct.error) as float_exc:
            record["issues"].append(f"unreadable: {float_exc}")
            return record
    except (EOFError, OSError) as exc:
        record["issues"].append(f"unreadable: {exc}")
        return record

    if duration < args.min_duration:
        record["issues"].append(f"too short ({duration:.2f}s)")
    if duration > args.max_duration:
        record["issues"].append(f"too long ({duration:.2f}s)")
    if peak is not None:
        if peak < args.silence_peak:
            record["issues"].append(f"near-silent (peak {peak:.4f})")
        elif peak >= 0.999:
            record["issues"].append("possible clipping (peak at full scale)")
    return record


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.manifest)

    results = []
    missing = 0
    for row in rows:
        audio_path = Path(row["audio_path"])
        result = {
            "utterance_id": row.get("utterance_id"),
            "tts_system_id": row.get("tts_system_id"),
            "preprocessing_method": row.get("preprocessing_method"),
            "scenario": row.get("scenario"),
            "audio_path": str(audio_path),
        }
        if not audio_path.exists():
            missing += 1
            result["issues"] = ["missing"]
            results.append(result)
            continue
        result.update(inspect_wav(audio_path, args))

        expected_rate = (row.get("metadata") or {}).get("sample_rate_hz")
        if expected_rate and result.get("sample_rate_hz") not in (None, expected_rate):
            result["issues"].append(
                f"sample rate {result['sample_rate_hz']} != configured {expected_rate}"
            )
        results.append(result)

    write_jsonl(args.output, results)

    # Per-system aggregates: duration stats, sample rates seen, issue counts.
    by_system: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        by_system[str(result.get("tts_system_id"))].append(result)

    summary: dict = {
        "total": len(results),
        "missing": missing,
        "with_issues": sum(1 for r in results if r.get("issues")),
        "by_system": {},
    }
    for system, system_rows in sorted(by_system.items()):
        durations = [
            r["duration_s"] for r in system_rows if isinstance(r.get("duration_s"), float)
        ]
        summary["by_system"][system] = {
            "count": len(system_rows),
            "missing": sum(1 for r in system_rows if r.get("issues") == ["missing"]),
            "with_issues": sum(1 for r in system_rows if r.get("issues")),
            "sample_rates": sorted(
                {r["sample_rate_hz"] for r in system_rows if r.get("sample_rate_hz")}
            ),
            "duration_s": {
                "mean": round(sum(durations) / len(durations), 2) if durations else None,
                "min": min(durations) if durations else None,
                "max": max(durations) if durations else None,
            },
        }

    summary_path = Path(args.output).with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"wrote {args.output}")
    print(f"wrote {summary_path}")
    print(
        f"total={summary['total']} missing={summary['missing']} "
        f"with_issues={summary['with_issues']}"
    )
    for result in results:
        if result.get("issues") and result["issues"] != ["missing"]:
            print(f"  {result['utterance_id']}: {'; '.join(result['issues'])}")
    return 0 if summary["with_issues"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
