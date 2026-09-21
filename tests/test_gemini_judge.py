import json

from greekttsbench.gemini_judge import (
    GeminiAudioJudge,
    audio_mime_type,
    response_text,
    response_usage_metadata,
)


def test_audio_mime_type_maps_wav():
    assert audio_mime_type("clip.wav") == "audio/wav"


def test_batch_request_embeds_inline_audio_and_reference(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake wav bytes")
    judge = GeminiAudioJudge(
        model="gemini-3.5-flash",
        api="vertex",
        dimensions=["naturalness"],
        use_reference=True,
    )

    row = judge.batch_request("utt-1", audio, "Γεια σου κόσμε")
    payload = json.dumps(row, ensure_ascii=False)

    assert row["key"] == "utt-1"
    assert row["request"]["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "audio/wav"
    assert "Γεια σου κόσμε" in payload
    assert "fake wav bytes" not in payload
    assert "GEMINI_API_KEY" not in payload


def test_batch_request_can_reference_gcs_audio_uri(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake wav bytes")
    judge = GeminiAudioJudge(model="gemini-3.5-flash", api="vertex")

    row = judge.batch_request(
        "utt-1",
        audio,
        "Γεια σου κόσμε",
        audio_uri="gs://bucket/artifacts/audio/clip.wav",
    )

    part = row["request"]["contents"][0]["parts"][0]
    assert part["file_data"]["file_uri"] == "gs://bucket/artifacts/audio/clip.wav"
    assert "inline_data" not in part


def test_generation_config_uses_system_instruction_without_credentials():
    judge = GeminiAudioJudge(api="vertex", dimensions=["naturalness"], prompt_style="strict")
    config = judge.generation_config()

    assert config["temperature"] == 0.0
    assert config["response_mime_type"] == "application/json"
    assert "flaws_and_deductions_log" in config["system_instruction"]


def test_response_text_reads_rest_style_dict():
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": '{"scores": {"naturalness": 4, "overall_mos": 4}}'}
                    ]
                }
            }
        ]
    }

    assert response_text(payload).startswith('{"scores"')


def test_response_usage_metadata_reads_rest_style_dict():
    payload = {"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}

    assert response_usage_metadata(payload) == {
        "promptTokenCount": 10,
        "candidatesTokenCount": 5,
    }
