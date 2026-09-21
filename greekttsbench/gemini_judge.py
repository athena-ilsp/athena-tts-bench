"""Gemini audio-in MOS judge backend.

This module keeps the hosted Gemini path separate from the offline Qwen judge
so API-key handling never leaks into the HPC-oriented runner.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Optional

from greekttsbench.mos_judge import (
    DEFAULT_DIMENSIONS,
    build_system_prompt,
    build_user_prompt,
    parse_mos_response,
)


AUDIO_MIME_BY_SUFFIX = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


def audio_mime_type(path: str | Path) -> str:
    """Return a Gemini-supported audio MIME type for a local audio file."""
    suffix = Path(path).suffix.lower()
    try:
        return AUDIO_MIME_BY_SUFFIX[suffix]
    except KeyError as exc:
        raise ValueError(f"unsupported audio suffix for Gemini input: {suffix!r}") from exc


def first_env(names: list[str]) -> Optional[str]:
    """Return the first non-empty environment value from a list of names."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def response_text(response: Any) -> str:
    """Extract generated text from a Gen AI SDK response or REST-style dict."""
    if response is None:
        return ""

    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text.strip()

    if isinstance(response, dict):
        return _text_from_response_dict(response)

    candidates = getattr(response, "candidates", None) or []
    parts: list[Any] = []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = list(getattr(content, "parts", None) or [])
    chunks: list[str] = []
    for part in parts:
        part_text = getattr(part, "text", None)
        if isinstance(part_text, str):
            chunks.append(part_text)
    return "\n".join(chunks).strip()


def response_usage_metadata(response: Any) -> Optional[dict[str, Any]]:
    """Extract SDK/REST usage metadata, if the provider returned it."""
    if response is None:
        return None
    if isinstance(response, dict):
        usage = response.get("usageMetadata") or response.get("usage_metadata")
        return usage if isinstance(usage, dict) else None

    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json", exclude_none=True)
    if hasattr(usage, "to_json_dict"):
        return usage.to_json_dict()
    if hasattr(usage, "__dict__"):
        return {
            key: value
            for key, value in vars(usage).items()
            if not key.startswith("_") and value is not None
        }
    return None


def _text_from_response_dict(response: dict[str, Any]) -> str:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        return ""
    chunks = [part.get("text") for part in parts if isinstance(part, dict)]
    return "\n".join(chunk for chunk in chunks if isinstance(chunk, str)).strip()


class GeminiAudioJudge:
    """Score TTS audio with Gemini's audio-understanding API."""

    def __init__(
        self,
        *,
        model: str = "gemini-3.5-flash",
        api: str = "developer",
        api_key_env: str = "GEMINI_API_KEY",
        location: Optional[str] = None,
        project_envs: Optional[list[str]] = None,
        location_envs: Optional[list[str]] = None,
        default_location: str = "us-central1",
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
        seed: Optional[int] = 0,
        use_reference: bool = True,
        dimensions: Optional[list[str]] = None,
        scale_min: float = 1.0,
        scale_max: float = 5.0,
        prompt_style: str = "strict",
        model_family: str = "gemini",
        response_mime_type: str = "application/json",
    ):
        self.model = model
        self.api = api
        self.api_key_env = api_key_env
        self.location = location
        self.project_envs = project_envs or ["VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"]
        self.location_envs = location_envs or ["VERTEX_LOCATION", "GOOGLE_CLOUD_LOCATION"]
        self.default_location = default_location
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.seed = seed
        self.use_reference = use_reference
        self.dimensions = dimensions or list(DEFAULT_DIMENSIONS)
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.prompt_style = prompt_style
        self.model_family = model_family
        self.response_mime_type = response_mime_type
        self.system_prompt = build_system_prompt(
            self.dimensions, use_reference, style=prompt_style
        )

        self._client = None
        self._types = None

    def load(self) -> None:
        """Create the Gemini SDK client (idempotent)."""
        if self._client is not None:
            return
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Gemini judge backend needs google-genai installed in the local "
                "environment: ./.venv/bin/python -m pip install google-genai"
            ) from exc

        if self.api in {"vertex", "vertex_ai"}:
            project = first_env(self.project_envs)
            location = self.location or first_env(self.location_envs) or self.default_location
            if not project:
                names = " or ".join(self.project_envs)
                raise RuntimeError(
                    f"Vertex AI Gemini needs {names} set in the local environment."
                )
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
        elif self.api in {"developer", "developer_api", "ai_studio"}:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Gemini API key not found: set {self.api_key_env} in the local "
                    "environment. Do not put the key in repo config or send it to HPC."
                )
            self._client = genai.Client(api_key=api_key)
        else:
            raise ValueError(f"unsupported Gemini API mode: {self.api!r}")
        self._types = types

    def close(self) -> None:
        client = self._client
        if client is not None and hasattr(client, "close"):
            client.close()
        self._client = None

    def generation_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "system_instruction": self.system_prompt,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "candidate_count": 1,
        }
        if self.seed is not None:
            config["seed"] = self.seed
        if self.response_mime_type:
            config["response_mime_type"] = self.response_mime_type
        return config

    def _sync_contents(self, audio_path: str | Path, reference_text: Optional[str]) -> Any:
        types = self._types
        audio_path = Path(audio_path)
        prompt = build_user_prompt(reference_text if self.use_reference else None)
        return [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        data=audio_path.read_bytes(),
                        mime_type=audio_mime_type(audio_path),
                    ),
                    types.Part(text=prompt),
                ],
            )
        ]

    def score_audio(
        self, audio_path: str | Path, reference_text: Optional[str] = None
    ) -> dict[str, Any]:
        """Score one local audio clip and return parsed MOS ratings."""
        self.load()
        types = self._types
        response = self._client.models.generate_content(
            model=self.model,
            contents=self._sync_contents(audio_path, reference_text),
            config=types.GenerateContentConfig(**self.generation_config()),
        )
        raw = response_text(response)
        parsed = parse_mos_response(
            raw,
            self.dimensions,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
        )
        parsed["raw_response"] = raw
        parsed["usage_metadata"] = response_usage_metadata(response)
        return parsed

    def batch_request(
        self,
        key: str,
        audio_path: str | Path,
        reference_text: Optional[str] = None,
        audio_uri: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build one JSONL request row for the Gemini Batch API."""
        audio_path = Path(audio_path)
        prompt = build_user_prompt(reference_text if self.use_reference else None)
        if audio_uri:
            audio_part = {
                "file_data": {
                    "mime_type": audio_mime_type(audio_path),
                    "file_uri": audio_uri,
                }
            }
        else:
            audio_part = {
                "inline_data": {
                    "mime_type": audio_mime_type(audio_path),
                    "data": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
                }
            }
        return {
            "key": key,
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            audio_part,
                            {"text": prompt},
                        ],
                    }
                ],
                "system_instruction": {"parts": [{"text": self.system_prompt}]},
                "generation_config": {
                    "max_output_tokens": self.max_output_tokens,
                    "temperature": self.temperature,
                    "candidate_count": 1,
                    **({"seed": self.seed} if self.seed is not None else {}),
                    **(
                        {"response_mime_type": self.response_mime_type}
                        if self.response_mime_type
                        else {}
                    ),
                },
            },
        }
