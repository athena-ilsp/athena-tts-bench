"""LLM-as-a-judge MOS scoring for synthesized Greek speech.

This module asks a multimodal (audio-in) LLM to listen to a synthesized WAV and
return MOS-style ratings. The judge runs Qwen2.5-Omni in-process via
transformers, so it needs no network or API key and works on offline HPC nodes
(e.g. Leonardo compute nodes). Point ``model_path`` at a local weight snapshot.

The LLM MOS is an automatic proxy for human MOS, not a replacement. Validate its
correlation against human ratings on a pilot subset before reporting it.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


# ── Rating dimensions ──────────────────────────────────────────────────────
# These mirror the human-evaluation dimensions in docs/evaluation_protocol.md.

DEFAULT_DIMENSIONS = ["naturalness", "intelligibility", "pronunciation", "prosody"]

DIMENSION_DESCRIPTIONS = {
    "naturalness": "how natural and human-like the voice sounds (vs. robotic or artificial)",
    "intelligibility": "how clearly and effortlessly the words can be understood",
    "pronunciation": "correctness of Greek pronunciation, including numbers, dates, acronyms and any foreign words, judged against the reference text",
    "prosody": "appropriateness of rhythm, stress, intonation and pausing for the sentence",
}

# No-reference variant: the judge is not shown the target text, so the
# pronunciation criterion cannot lean on it.
DIMENSION_DESCRIPTIONS_NO_REFERENCE = {
    **DIMENSION_DESCRIPTIONS,
    "pronunciation": "correctness and clarity of Greek pronunciation, including any numbers, acronyms or foreign words you hear",
}

_REFERENCE_NOTE = (
    "The reference text the system was asked to read is provided so you can judge\n"
    "whether numbers, dates and acronyms were pronounced correctly and completely.\n"
)
_NO_REFERENCE_NOTE = "You are not given the target text; judge only what you hear.\n"


# ── Judge prompt ───────────────────────────────────────────────────────────

MOS_SYSTEM_PROMPT = """You are an expert evaluator of Greek text-to-speech (TTS) audio.

You will listen to a short synthesized speech clip in Greek and rate its quality
on a Mean Opinion Score (MOS) scale from 1 to 5, where:
  1 = bad, 2 = poor, 3 = fair, 4 = good, 5 = excellent.
Half-point scores (e.g. 3.5) are allowed.

Rate each of the following dimensions independently:
{dimensions}

Also give a single "overall" MOS score for the clip.

Be a strict, calibrated judge: reserve 5 for genuinely natural, fully
intelligible speech with correct Greek pronunciation and appropriate prosody.
{reference_note}
Respond with ONLY a single JSON object and no other text, in exactly this shape:
{{
{score_fields}
  "overall": <number 1-5>,
  "comment": "<one short sentence in English explaining the main issue, or 'none'>"
}}"""


# ── Strict "flaw-first" judge prompt (negative gatekeeping) ─────────────────
# The standard prompt above tends to drift toward 4-5 for almost every clip
# (leniency bias). This variant forces the model to (1) log every audible flaw
# first, then (2) start each dimension at 5.0 and subtract per logged flaw, so a
# perfect 5.0 has to be *earned* rather than handed out by default. It is the
# user-supplied "flaw-first" layout, re-pointed at what the model HEARS — the
# judge listens to a WAV, so it grades audible artifacts, not text structure.

STRICT_MOS_SYSTEM_PROMPT = """You are a severe, highly critical, explicitly calibrated judge of Greek text-to-speech (TTS) AUDIO. Your job is to detect flaws. Act as a pessimistic grader: assume the system has FAILED to sound human until the audio proves otherwise.

You will LISTEN to a short synthesized Greek speech clip and rate it. Judge only what you actually hear.

### SCORING PHILOSOPHY & ANCHORS
Score on a Mean Opinion Score (MOS) from 1.0 to 5.0 (half points like 3.5 or 4.5 are allowed).
  - 5.0 is an absolute ceiling: reserved ONLY for audio with zero artifacts, completely natural human delivery, flawless Greek pronunciation, and intuitive rhythm. Almost nothing earns it.
  - 4.0 is "acceptable but clearly synthetic": ordinary, usable TTS.
  - 3.0 or lower: audible defects — robotic timbre, mispronounced numbers/dates/acronyms, garbled or colliding words, wrong stress, or unnatural pausing.
Do NOT give a high score just because the clip is intelligible; intelligible-but-robotic is a 3-4, not a 5.

### DIMENSION EVALUATION CRITERIA
{dimensions}
  CRITICAL PENALTY: deduct 1.5 points immediately if you hear a number, date or year read wrong, left unexpanded, or spoken in the wrong language instead of full grammatical Greek (e.g. "1995" must sound like "χίλια εννιακόσια ενενήντα πέντε"). Deduct heavily if an acronym such as "ΔΕΗ" is run together as a word instead of spelled out letter-by-letter ("Δε-Έ-Η").
{reference_note}
### STEP-BY-STEP WORKFLOW (follow in this exact order to prevent leniency bias)
Step 1: Identify EVERY audible flaw — artifacts, mispronunciations, awkward phrasing, wrong stress, bad pausing. List them explicitly under each dimension.
Step 2: Start each dimension score at 5.0 and subtract for each flaw you logged in Step 1.
Step 3: Output ONLY the final JSON object below — no preamble, no markdown fences, no text outside the JSON.

### OUTPUT FORMAT
Your response must be exclusively a single JSON object in exactly this shape:
{{
  "flaws_and_deductions_log": {{
{flaw_fields}
  }},
  "scores": {{
{score_fields}
    "overall_mos": <number 1.0-5.0>
  }}
}}"""


def build_system_prompt(
    dimensions: list[str], use_reference: bool = True, style: str = "standard"
) -> str:
    """Render the judge system prompt for the requested dimensions.

    ``style`` selects the prompt layout: ``"standard"`` (calibrated but lenient)
    or ``"strict"`` (flaw-first / negative gatekeeping, to counter the score
    inflation seen in the pilot).
    """
    descriptions = (
        DIMENSION_DESCRIPTIONS if use_reference else DIMENSION_DESCRIPTIONS_NO_REFERENCE
    )
    bullet_lines = "\n".join(
        f"  - {dim}: {descriptions.get(dim, dim)}" for dim in dimensions
    )
    reference_note = _REFERENCE_NOTE if use_reference else _NO_REFERENCE_NOTE

    if style == "strict":
        flaw_fields = "\n".join(
            f'    "{dim}_anomalies": "List flaws or write \'None\'",'
            for dim in dimensions
        )
        score_fields = "\n".join(f'    "{dim}": <number 1.0-5.0>,' for dim in dimensions)
        return STRICT_MOS_SYSTEM_PROMPT.format(
            dimensions=bullet_lines,
            flaw_fields=flaw_fields,
            score_fields=score_fields,
            reference_note=reference_note,
        )

    score_fields = "\n".join(f'  "{dim}": <number 1-5>,' for dim in dimensions)
    return MOS_SYSTEM_PROMPT.format(
        dimensions=bullet_lines,
        score_fields=score_fields,
        reference_note=reference_note,
    )


def build_user_prompt(reference_text: Optional[str]) -> str:
    """Render the per-clip user prompt."""
    if reference_text:
        return (
            "Reference text the TTS system was asked to read (in Greek):\n"
            f"“{reference_text}”\n\n"
            "Listen to the attached audio and return the JSON ratings."
        )
    return "Listen to the attached audio and return the JSON ratings."


# ── Response parsing ───────────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def parse_mos_response(
    text: str,
    dimensions: list[str],
    *,
    scale_min: float = 1.0,
    scale_max: float = 5.0,
) -> dict[str, Any]:
    """Parse a judge response into structured, clamped MOS scores.

    Returns a dict with keys: mos (per-dimension), mos_overall, comment,
    parse_ok. On failure parse_ok is False and numeric fields are None.
    """
    result: dict[str, Any] = {
        "mos": {dim: None for dim in dimensions},
        "mos_overall": None,
        "comment": None,
        "flaws": None,
        "parse_ok": False,
    }
    if not text:
        return result

    match = _JSON_RE.search(text)
    if not match:
        return result
    try:
        payload = json.loads(match.group(0))
    except (ValueError, TypeError):
        return result
    if not isinstance(payload, dict):
        return result

    def coerce(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return _clamp(float(value), scale_min, scale_max)
        if isinstance(value, str):
            number = re.search(r"-?\d+(?:\.\d+)?", value)
            if number:
                return _clamp(float(number.group(0)), scale_min, scale_max)
        return None

    # The strict "flaw-first" prompt nests scores under "scores" and names the
    # overall "overall_mos"; the standard prompt puts them at the top level. Read
    # whichever is present so both prompt styles parse.
    scores_obj = payload.get("scores")
    score_source = scores_obj if isinstance(scores_obj, dict) else payload

    dim_scores = {dim: coerce(score_source.get(dim)) for dim in dimensions}
    result["mos"] = dim_scores

    overall = coerce(score_source.get("overall"))
    if overall is None:
        overall = coerce(score_source.get("overall_mos"))
    if overall is None:
        present = [v for v in dim_scores.values() if v is not None]
        if present:
            overall = round(sum(present) / len(present), 3)
    result["mos_overall"] = overall

    comment = payload.get("comment")
    if isinstance(comment, str):
        result["comment"] = comment.strip()

    flaws = payload.get("flaws_and_deductions_log")
    if isinstance(flaws, dict):
        result["flaws"] = flaws

    result["parse_ok"] = overall is not None
    return result


# ── Judge (in-process Qwen2.5-Omni via transformers) ───────────────────────


class QwenOmniLocalJudge:
    """Score TTS audio with a self-hosted Qwen2.5-Omni model via transformers.

    The model is loaded once and reused across clips. Only text output is used,
    so the speech generator (talker) is disabled to save memory. No network or
    API key is required, which suits offline HPC nodes; set ``model_path`` to a
    local snapshot when compute nodes have no internet access.
    """

    def __init__(
        self,
        *,
        model_path: str = "Qwen/Qwen2.5-Omni-7B",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        attn_implementation: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        use_audio_in_video: bool = False,
        use_reference: bool = True,
        dimensions: Optional[list[str]] = None,
        scale_min: float = 1.0,
        scale_max: float = 5.0,
        prompt_style: str = "standard",
        model_family: str = "qwen2_5_omni",
    ):
        self.model_family = model_family
        self.model_path = model_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_audio_in_video = use_audio_in_video
        self.use_reference = use_reference
        self.dimensions = dimensions or list(DEFAULT_DIMENSIONS)
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.prompt_style = prompt_style
        self.system_prompt = build_system_prompt(
            self.dimensions, use_reference, style=prompt_style
        )

        self._model = None
        self._processor = None
        self._process_mm_info = None
        self._torch = None

    def load(self) -> None:
        """Load the model and processor (idempotent)."""
        if self._model is not None:
            return
        try:
            import torch
            from qwen_omni_utils import process_mm_info

            if self.model_family == "qwen3_omni":
                # Qwen3-Omni is an MoE (30B total / ~3B active) audio-in model;
                # different transformers classes from Qwen2.5-Omni.
                from transformers import (
                    Qwen3OmniMoeForConditionalGeneration as OmniModel,
                    Qwen3OmniMoeProcessor as OmniProcessor,
                )
            else:
                from transformers import (
                    Qwen2_5OmniForConditionalGeneration as OmniModel,
                    Qwen2_5OmniProcessor as OmniProcessor,
                )
        except ImportError as exc:
            raise RuntimeError(
                "The transformers judge backend needs: transformers (with "
                f"{self.model_family} support), qwen-omni-utils, torch, accelerate, "
                "soundfile. Qwen3-Omni needs a newer transformers than Qwen2.5-Omni; "
                "install it (ideally in a separate env) or set up the node."
            ) from exc

        kwargs: dict[str, Any] = {
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        model = OmniModel.from_pretrained(self.model_path, **kwargs)
        # We only consume text; free the speech-generation head if present.
        if hasattr(model, "disable_talker"):
            model.disable_talker()
        model.eval()

        self._model = model
        self._processor = OmniProcessor.from_pretrained(self.model_path)
        self._process_mm_info = process_mm_info
        self._torch = torch

    def _generate(self, conversation: list[dict[str, Any]]) -> str:
        self.load()
        processor = self._processor
        torch = self._torch

        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = self._process_mm_info(
            conversation, use_audio_in_video=self.use_audio_in_video
        )
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )
        inputs = inputs.to(self._model.device).to(self._model.dtype)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "return_audio": False,
            "use_audio_in_video": self.use_audio_in_video,
        }
        if self.temperature and self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output = self._model.generate(**inputs, **gen_kwargs)
        if isinstance(output, (tuple, list)):
            output = output[0]

        prompt_len = inputs["input_ids"].shape[1]
        trimmed = output[:, prompt_len:]
        decoded = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0].strip() if decoded else ""

    def score_audio(
        self, audio_path: str | Path, reference_text: Optional[str] = None
    ) -> dict[str, Any]:
        """Score a single WAV and return structured MOS ratings.

        The returned dict includes parsed scores plus the raw response text so
        the run is auditable. ``reference_text`` is ignored when the judge was
        built with ``use_reference=False``.
        """
        effective_reference = reference_text if self.use_reference else None
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": build_user_prompt(effective_reference)},
                ],
            },
        ]
        raw = self._generate(conversation)
        parsed = parse_mos_response(
            raw,
            self.dimensions,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
        )
        parsed["raw_response"] = raw
        return parsed


# ── Aggregation ────────────────────────────────────────────────────────────


def _mean(values: list[float]) -> Optional[float]:
    return round(sum(values) / len(values), 3) if values else None


def _aggregate(rows: list[dict[str, Any]], dimensions: list[str]) -> dict[str, Any]:
    scored = [r for r in rows if r.get("mos_overall") is not None]
    overall = _mean([r["mos_overall"] for r in scored])
    per_dim = {}
    for dim in dimensions:
        vals = [
            r["mos"][dim]
            for r in scored
            if isinstance(r.get("mos"), dict) and r["mos"].get(dim) is not None
        ]
        per_dim[dim] = _mean(vals)
    return {
        "count": len(rows),
        "scored": len(scored),
        "mean_overall_mos": overall,
        "mean_dimension_mos": per_dim,
    }


def summarize_mos_scores(
    rows: list[dict[str, Any]], dimensions: list[str]
) -> dict[str, Any]:
    """Aggregate MOS rows overall and by system, system+method, and scenario."""

    def group(key_fn) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[key_fn(row)].append(row)
        return {
            key: _aggregate(group_rows, dimensions)
            for key, group_rows in sorted(buckets.items())
        }

    return {
        "overall": _aggregate(rows, dimensions),
        "by_system": group(lambda r: str(r.get("tts_system_id"))),
        "by_system_method": group(
            lambda r: f"{r.get('tts_system_id')}::{r.get('preprocessing_method')}"
        ),
        "by_scenario": group(lambda r: str(r.get("scenario"))),
    }
