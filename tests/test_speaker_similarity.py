from pathlib import Path

from greekttsbench.speaker_similarity import (
    SpeakerSimilarityScorer,
    resolve_audio_path,
    resolve_reference_audio,
    summarize_speaker_similarity_scores,
)


def test_scorer_construction_needs_no_speechbrain():
    # Heavy ML imports happen in load(); dry-runs and config validation should
    # work in lightweight environments.
    scorer = SpeakerSimilarityScorer()
    assert scorer.model_source == "speechbrain/spkrec-ecapa-voxceleb"


def test_reference_resolution_prefers_system_over_voice():
    row = {
        "tts_system_id": "sys_a",
        "tts_family": "fam",
        "metadata": {"voice": "speaker 0"},
    }
    config = {
        "reference_audio_by_system": {"sys_a": "refs/sys_a.wav"},
        "reference_audio_by_family_voice": {
            "fam::speaker 0": "refs/fam_speaker0.wav"
        },
        "reference_audio_by_voice": {"speaker 0": "refs/speaker0.wav"},
    }

    path, selector = resolve_reference_audio(row, config)

    assert path == "refs/sys_a.wav"
    assert selector == "system:sys_a"


def test_reference_resolution_supports_family_voice_before_global_voice():
    row = {
        "tts_system_id": "sys_b",
        "tts_family": "chatterbox",
        "metadata": {"voice": "data/refs/male.wav"},
    }
    config = {
        "reference_audio_by_family_voice": {
            "chatterbox::data/refs/male.wav": "refs/chatterbox_male.wav"
        },
        "reference_audio_by_voice": {"data/refs/male.wav": "refs/global_male.wav"},
    }

    path, selector = resolve_reference_audio(row, config)

    assert path == "refs/chatterbox_male.wav"
    assert selector == "family_voice:chatterbox::data/refs/male.wav"


def test_reference_resolution_missing_is_explicit():
    path, selector = resolve_reference_audio(
        {"tts_system_id": "sys_without_ref", "metadata": {"voice": "voice"}},
        {"reference_audio_by_system": {}},
    )

    assert path is None
    assert selector is None


def test_resolve_audio_path_prefers_cwd_then_config_dir(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    config_dir = tmp_path / "configs"
    cwd.mkdir()
    config_dir.mkdir()
    cwd_ref = cwd / "refs" / "voice.wav"
    cwd_ref.parent.mkdir()
    cwd_ref.write_bytes(b"placeholder")
    config_ref = config_dir / "other.wav"
    config_ref.write_bytes(b"placeholder")

    monkeypatch.chdir(cwd)

    assert resolve_audio_path("refs/voice.wav", config_dir=config_dir) == Path(
        "refs/voice.wav"
    )
    assert resolve_audio_path("other.wav", config_dir=config_dir) == config_ref


def test_summarize_speaker_similarity_scores_groups_and_excludes_unscored():
    rows = [
        {
            "tts_system_id": "vits_el",
            "preprocessing_method": "raw",
            "scenario": "numbers_dates",
            "speaker_similarity": 0.8,
        },
        {
            "tts_system_id": "vits_el",
            "preprocessing_method": "regex",
            "scenario": "numbers_dates",
            "speaker_similarity": 0.6,
        },
        {
            "tts_system_id": "vits_el",
            "preprocessing_method": "raw",
            "scenario": "acronyms",
            "speaker_similarity": None,
        },
    ]

    summary = summarize_speaker_similarity_scores(rows)

    assert summary["overall"]["count"] == 3
    assert summary["overall"]["scored"] == 2
    assert summary["overall"]["mean_speaker_similarity"] == 0.7
    assert summary["overall"]["min_speaker_similarity"] == 0.6
    assert summary["overall"]["max_speaker_similarity"] == 0.8
    assert summary["by_system"]["vits_el"]["mean_speaker_similarity"] == 0.7
    assert (
        summary["by_system_method"]["vits_el::regex"]["mean_speaker_similarity"]
        == 0.6
    )
    assert summary["by_scenario"]["acronyms"]["scored"] == 0
    assert summary["by_scenario"]["acronyms"]["mean_speaker_similarity"] is None
