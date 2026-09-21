"""UTMOS objective naturalness prediction for synthesized speech.

UTMOS (the UTokyo-SaruLab MOS prediction system, winner of the VoiceMOS
Challenge 2022 main track) is a wav2vec2-based network that predicts a 1-5
naturalness MOS directly from the waveform — no reference text, no ASR, no
judge prompt. This module wraps the SpeechMOS packaging of the strong learner
(``utmos22_strong``) loaded through torch.hub. The model runs in-process and,
once the torch.hub cache is pre-populated, needs no network, which suits
offline HPC nodes (e.g. Leonardo compute nodes).

Caveats for this benchmark: UTMOS was trained on English MOS ratings (BVCC),
so Greek clips are out of domain. Use it for relative comparisons between
systems and preprocessing conditions, not for absolute quality claims, and do
not lean on it for the Greek-vs-English analysis (English is in-domain for
UTMOS, so it favors English by construction). It complements — never replaces
— the LLM-judge and human MOS tracks.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


DEFAULT_HUB_REPO = "tarepan/SpeechMOS:v1.2.0"
DEFAULT_ENTRYPOINT = "utmos22_strong"


class UTMOSPredictor:
    """Score waveforms with a UTMOS model loaded via torch.hub.

    The model is loaded once and reused across clips. ``source="github"``
    resolves ``repo`` as a hub spec (cached under ``$TORCH_HOME/hub`` after the
    first call, then reused without network); ``source="local"`` resolves it as
    a path to a SpeechMOS clone for fully offline nodes.
    """

    def __init__(
        self,
        *,
        repo: str = DEFAULT_HUB_REPO,
        entrypoint: str = DEFAULT_ENTRYPOINT,
        source: str = "github",
        device: str = "auto",
    ):
        self.repo = repo
        self.entrypoint = entrypoint
        self.source = source
        self.device = device

        self._model = None
        self._torch = None
        self._soundfile = None
        self._resolved_device: Optional[str] = None

    def load(self) -> None:
        """Load the predictor and move it to the requested device (idempotent)."""
        if self._model is not None:
            return
        try:
            import soundfile
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "The UTMOS predictor needs: torch, torchaudio, soundfile. "
                "Install them, or on offline nodes pre-cache the model per "
                "configs/utmos.json notes."
            ) from exc

        model = torch.hub.load(
            self.repo, self.entrypoint, source=self.source, trust_repo=True
        )
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        self._model = model
        self._torch = torch
        self._soundfile = soundfile
        self._resolved_device = device

    def score_audio(self, audio_path: str | Path) -> float:
        """Predict the UTMOS score for one audio file.

        Raises on unreadable/empty audio; callers doing batch runs should catch
        per clip and record the failure instead of aborting the job.
        """
        self.load()
        torch = self._torch

        # always_2d gives (frames, channels); average channels to mono. The
        # model resamples to its 16 kHz input rate internally.
        data, sample_rate = self._soundfile.read(
            str(audio_path), dtype="float32", always_2d=True
        )
        if data.shape[0] == 0:
            raise ValueError(f"empty audio: {audio_path}")
        wave = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)

        with torch.no_grad():
            score = self._model(wave.to(self._resolved_device), sample_rate)
        return round(float(score.squeeze().item()), 4)


# ── Aggregation ────────────────────────────────────────────────────────────


def _mean(values: list[float]) -> Optional[float]:
    return round(sum(values) / len(values), 3) if values else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [r["utmos"] for r in rows if r.get("utmos") is not None]
    return {
        "count": len(rows),
        "scored": len(values),
        "mean_utmos": _mean(values),
        "min_utmos": round(min(values), 3) if values else None,
        "max_utmos": round(max(values), 3) if values else None,
    }


def summarize_utmos_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate UTMOS rows overall and by system, system+method, and scenario."""

    def group(key_fn) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[key_fn(row)].append(row)
        return {
            key: _aggregate(group_rows) for key, group_rows in sorted(buckets.items())
        }

    return {
        "overall": _aggregate(rows),
        "by_system": group(lambda r: str(r.get("tts_system_id"))),
        "by_system_method": group(
            lambda r: f"{r.get('tts_system_id')}::{r.get('preprocessing_method')}"
        ),
        "by_scenario": group(lambda r: str(r.get("scenario"))),
    }
