import argparse
import struct
from pathlib import Path

from scripts.check_audio import inspect_wav


def write_float_wav(path: Path, samples: list[float], sample_rate: int = 16000) -> None:
    channels = 1
    bits_per_sample = 32
    sample_width = bits_per_sample // 8
    block_align = channels * sample_width
    byte_rate = sample_rate * block_align
    data = b"".join(struct.pack("<f", sample) for sample in samples)
    fmt = struct.pack(
        "<HHIIHH",
        3,  # WAVE_FORMAT_IEEE_FLOAT
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    riff_size = 4 + (8 + len(fmt)) + (8 + len(data))
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


def test_check_audio_reads_ieee_float_wav(tmp_path):
    wav_path = tmp_path / "float.wav"
    write_float_wav(wav_path, [-0.5, 0.0, 0.25, 0.75])
    args = argparse.Namespace(
        min_duration=0.0,
        max_duration=1.0,
        silence_peak=0.01,
    )

    result = inspect_wav(wav_path, args)

    assert result["issues"] == []
    assert result["wav_format"] == "ieee_float"
    assert result["sample_rate_hz"] == 16000
    assert result["sample_width_bytes"] == 4
    assert result["peak_amplitude"] == 0.75
