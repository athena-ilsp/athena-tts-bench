"""Speaker similarity scoring against explicit voice reference WAVs.

The benchmark texts come from Wikipedia, so they do not have meaningful
ground-truth speech. This module therefore compares each synthesized clip to a
user-provided reference WAV for the intended voice/system, using speaker
embeddings and cosine similarity. Reference selection is deliberately explicit:
callers must provide a mapping by TTS system or voice, and no synthesized or
dataset audio is used as a fallback reference.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


DEFAULT_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_SAVEDIR = "models/speechbrain-spkrec-ecapa-voxceleb"


def _as_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(path) for key, path in value.items() if path}


def resolve_reference_audio(
    row: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Return the reference WAV configured for a manifest row.

    Resolution order is most-specific first:

    1. ``row["reference_audio_path"]`` or ``row["reference_audio"]`` for
       manifests that already carry explicit reference paths.
    2. ``reference_audio_by_system[tts_system_id]``.
    3. ``reference_audio_by_family_voice["<tts_family>::<metadata.voice>"]``.
    4. ``reference_audio_by_voice[metadata.voice]``.

    The second returned value names the selector used, for auditability in
    result rows.
    """
    for row_key in ("reference_audio_path", "reference_audio"):
        value = row.get(row_key)
        if value:
            return str(value), f"row:{row_key}"

    by_system = _as_mapping(config.get("reference_audio_by_system"))
    system_id = str(row.get("tts_system_id") or "")
    if system_id and system_id in by_system:
        return by_system[system_id], f"system:{system_id}"

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    voice = metadata.get("voice") or row.get("voice")
    voice_key = str(voice) if voice is not None else ""
    family = str(row.get("tts_family") or "")

    by_family_voice = _as_mapping(config.get("reference_audio_by_family_voice"))
    family_voice_key = f"{family}::{voice_key}"
    if family and voice_key and family_voice_key in by_family_voice:
        return by_family_voice[family_voice_key], f"family_voice:{family_voice_key}"

    by_voice = _as_mapping(config.get("reference_audio_by_voice"))
    if voice_key and voice_key in by_voice:
        return by_voice[voice_key], f"voice:{voice_key}"

    return None, None


def resolve_audio_path(value: str | Path, *, config_dir: str | Path | None = None) -> Path:
    """Resolve a configured audio path without forcing absolute output paths.

    Relative paths are first interpreted from the current working directory,
    matching existing manifest behavior. If that path does not exist and a
    config directory is provided, the path is interpreted relative to that
    config file as a convenience for standalone configs.
    """
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists() or config_dir is None:
        return path
    return Path(config_dir) / path


class SpeakerSimilarityScorer:
    """Compute cosine similarity between SpeechBrain speaker embeddings."""

    def __init__(
        self,
        *,
        model_source: str = DEFAULT_MODEL_SOURCE,
        savedir: str | Path | None = DEFAULT_SAVEDIR,
        hparams_file: str | None = None,
        device: str = "auto",
    ):
        self.model_source = model_source
        self.savedir = str(savedir) if savedir is not None else None
        self.hparams_file = hparams_file
        self.device = device

        self._classifier = None
        self._torch = None
        self._resolved_device: Optional[str] = None
        self._reference_embeddings: dict[str, Any] = {}

    def load(self) -> None:
        """Load the embedding model (idempotent, imports heavy deps lazily)."""
        if self._classifier is not None:
            return

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Speaker similarity needs: speechbrain, torch, torchaudio. "
                "Install the optional speaker dependencies and pre-cache the "
                "SpeechBrain model for offline nodes."
            ) from exc

        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            try:
                from speechbrain.pretrained import EncoderClassifier
            except ImportError as exc:
                raise RuntimeError(
                    "Speaker similarity needs SpeechBrain. Install speechbrain "
                    "or use the greekttsbench[speaker] optional dependencies."
                ) from exc

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        kwargs: dict[str, Any] = {
            "source": self.model_source,
            "run_opts": {"device": device},
        }
        if self.savedir:
            kwargs["savedir"] = self.savedir
        if self.hparams_file:
            kwargs["hparams_file"] = self.hparams_file

        self._classifier = EncoderClassifier.from_hparams(**kwargs)
        self._torch = torch
        self._resolved_device = device

    def _load_audio_tensor(self, audio_path: str | Path):
        """Load audio as a mono tensor shaped ``[1, time]`` for encode_batch."""
        self.load()
        classifier = self._classifier
        torch = self._torch

        try:
            signal = classifier.load_audio(str(audio_path))
        except AttributeError:
            try:
                import torchaudio
            except ImportError as exc:
                raise RuntimeError(
                    "Speaker similarity fallback audio loading needs torchaudio."
                ) from exc

            signal, sample_rate = torchaudio.load(str(audio_path))
            target_sample_rate = int(getattr(classifier.hparams, "sample_rate", 16000))
            if sample_rate != target_sample_rate:
                signal = torchaudio.transforms.Resample(
                    sample_rate, target_sample_rate
                )(signal)
            if signal.ndim == 2 and signal.shape[0] > 1:
                signal = signal.mean(dim=0, keepdim=True)

        if signal.numel() == 0:
            raise ValueError(f"empty audio: {audio_path}")

        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        elif signal.ndim == 2:
            if signal.shape[0] == 1:
                pass
            elif signal.shape[1] == 1:
                signal = signal.transpose(0, 1)
            elif signal.shape[0] < signal.shape[1]:
                signal = signal.mean(dim=0, keepdim=True)
            else:
                signal = signal.mean(dim=1, keepdim=True).transpose(0, 1)
        else:
            raise ValueError(f"unsupported audio tensor shape for {audio_path}")

        return signal.to(self._resolved_device)

    def embedding(self, audio_path: str | Path):
        """Return a normalized speaker embedding for one audio file."""
        self.load()
        torch = self._torch
        signal = self._load_audio_tensor(audio_path)
        with torch.no_grad():
            embedding = self._classifier.encode_batch(signal).squeeze()
        norm = torch.linalg.vector_norm(embedding)
        if float(norm.item()) == 0.0:
            raise ValueError(f"zero speaker embedding for {audio_path}")
        return embedding / norm

    def reference_embedding(self, reference_path: str | Path):
        """Return a cached normalized embedding for a reference WAV."""
        key = str(Path(reference_path).expanduser().resolve(strict=False))
        if key not in self._reference_embeddings:
            self._reference_embeddings[key] = self.embedding(reference_path)
        return self._reference_embeddings[key]

    def similarity(self, audio_path: str | Path, reference_path: str | Path) -> float:
        """Return cosine similarity in speaker-embedding space."""
        generated = self.embedding(audio_path)
        reference = self.reference_embedding(reference_path)
        score = self._torch.nn.functional.cosine_similarity(
            generated.reshape(1, -1), reference.reshape(1, -1)
        )
        return round(float(score.squeeze().item()), 4)


def _mean(values: list[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["speaker_similarity"])
        for row in rows
        if isinstance(row.get("speaker_similarity"), (int, float))
    ]
    return {
        "count": len(rows),
        "scored": len(values),
        "mean_speaker_similarity": _mean(values),
        "min_speaker_similarity": round(min(values), 4) if values else None,
        "max_speaker_similarity": round(max(values), 4) if values else None,
    }


def summarize_speaker_similarity_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate speaker-similarity rows overall and by core dimensions."""

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
