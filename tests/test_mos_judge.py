from greekttsbench.mos_judge import (
    DEFAULT_DIMENSIONS,
    build_system_prompt,
    build_user_prompt,
    parse_mos_response,
    summarize_mos_scores,
)


def test_parse_mos_response_reads_clean_json():
    text = (
        '{"naturalness": 4, "intelligibility": 5, "pronunciation": 3.5, '
        '"prosody": 4, "overall": 4, "comment": "slight robotic tone"}'
    )
    result = parse_mos_response(text, DEFAULT_DIMENSIONS)
    assert result["parse_ok"] is True
    assert result["mos_overall"] == 4.0
    assert result["mos"]["pronunciation"] == 3.5
    assert result["comment"] == "slight robotic tone"


def test_parse_mos_response_extracts_json_from_surrounding_text():
    text = 'Here are the ratings:\n{"naturalness": 3, "overall": 3}\nThanks!'
    result = parse_mos_response(text, ["naturalness"])
    assert result["parse_ok"] is True
    assert result["mos_overall"] == 3.0


def test_parse_mos_response_clamps_out_of_range_scores():
    result = parse_mos_response('{"naturalness": 9, "overall": 0}', ["naturalness"])
    assert result["mos"]["naturalness"] == 5.0
    assert result["mos_overall"] == 1.0


def test_parse_mos_response_derives_overall_from_dimensions_when_missing():
    text = '{"naturalness": 4, "intelligibility": 2}'
    result = parse_mos_response(text, ["naturalness", "intelligibility"])
    assert result["mos_overall"] == 3.0


def test_parse_mos_response_handles_garbage():
    result = parse_mos_response("no json here", DEFAULT_DIMENSIONS)
    assert result["parse_ok"] is False
    assert result["mos_overall"] is None


def test_build_system_prompt_reference_modes_differ():
    with_ref = build_system_prompt(DEFAULT_DIMENSIONS, use_reference=True)
    no_ref = build_system_prompt(DEFAULT_DIMENSIONS, use_reference=False)
    assert "reference text" in with_ref
    assert "not given the target text" in no_ref
    assert "judged against the reference text" not in no_ref


def test_build_user_prompt_omits_reference_when_absent():
    assert "Reference text" not in build_user_prompt(None)
    assert "Reference text" in build_user_prompt("Γεια σου κόσμε")


def test_strict_prompt_is_flaw_first_and_has_log_fields():
    strict = build_system_prompt(DEFAULT_DIMENSIONS, style="strict")
    assert "flaws_and_deductions_log" in strict
    assert "absolute ceiling" in strict
    # flaw log key per dimension
    assert '"pronunciation_anomalies"' in strict
    # standard prompt does not use the flaw-first layout
    standard = build_system_prompt(DEFAULT_DIMENSIONS, style="standard")
    assert "flaws_and_deductions_log" not in standard


def test_parse_mos_response_reads_strict_nested_schema():
    text = (
        '{"flaws_and_deductions_log": {"pronunciation_errors": "read 1995 as digits"},'
        ' "scores": {"naturalness": 4, "intelligibility": 4.5,'
        ' "pronunciation": 2, "prosody": 3.5, "overall_mos": 3.0}}'
    )
    result = parse_mos_response(text, DEFAULT_DIMENSIONS)
    assert result["parse_ok"] is True
    assert result["mos_overall"] == 3.0
    assert result["mos"]["pronunciation"] == 2.0
    assert result["mos"]["intelligibility"] == 4.5
    assert isinstance(result["flaws"], dict)
    assert "pronunciation_errors" in result["flaws"]


def test_summarize_mos_scores_groups_by_system_and_scenario():
    rows = [
        {
            "tts_system_id": "vits_el",
            "preprocessing_method": "raw",
            "scenario": "numbers_dates",
            "mos_overall": 4.0,
            "mos": {"naturalness": 4.0},
        },
        {
            "tts_system_id": "vits_el",
            "preprocessing_method": "regex",
            "scenario": "numbers_dates",
            "mos_overall": 2.0,
            "mos": {"naturalness": 2.0},
        },
        {
            "tts_system_id": "vits_el",
            "preprocessing_method": "raw",
            "scenario": "acronyms",
            "mos_overall": None,
            "mos": {"naturalness": None},
        },
    ]
    summary = summarize_mos_scores(rows, ["naturalness"])
    assert summary["overall"]["scored"] == 2
    assert summary["overall"]["mean_overall_mos"] == 3.0
    assert summary["by_system"]["vits_el"]["mean_dimension_mos"]["naturalness"] == 3.0
    assert summary["by_scenario"]["acronyms"]["scored"] == 0
