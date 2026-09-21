from greekttsbench.utmos import UTMOSPredictor, summarize_utmos_scores


def test_predictor_construction_needs_no_torch():
    # torch/soundfile are imported lazily in load(); building the predictor
    # (e.g. for --dry-run) must work in environments without them.
    predictor = UTMOSPredictor()
    assert predictor.repo == "tarepan/SpeechMOS:v1.2.0"
    assert predictor.entrypoint == "utmos22_strong"


def test_summarize_utmos_scores_groups_and_excludes_unscored():
    rows = [
        {
            "tts_system_id": "vits_el",
            "preprocessing_method": "raw",
            "scenario": "numbers_dates",
            "utmos": 4.0,
        },
        {
            "tts_system_id": "vits_el",
            "preprocessing_method": "regex",
            "scenario": "numbers_dates",
            "utmos": 2.0,
        },
        {
            "tts_system_id": "vits_el",
            "preprocessing_method": "raw",
            "scenario": "acronyms",
            "utmos": None,
        },
    ]
    summary = summarize_utmos_scores(rows)
    assert summary["overall"]["count"] == 3
    assert summary["overall"]["scored"] == 2
    assert summary["overall"]["mean_utmos"] == 3.0
    assert summary["overall"]["min_utmos"] == 2.0
    assert summary["overall"]["max_utmos"] == 4.0
    assert summary["by_system"]["vits_el"]["mean_utmos"] == 3.0
    assert summary["by_system_method"]["vits_el::regex"]["mean_utmos"] == 2.0
    assert summary["by_scenario"]["acronyms"]["scored"] == 0
    assert summary["by_scenario"]["acronyms"]["mean_utmos"] is None


def test_summarize_utmos_scores_empty():
    summary = summarize_utmos_scores([])
    assert summary["overall"]["count"] == 0
    assert summary["overall"]["mean_utmos"] is None
    assert summary["by_system"] == {}
