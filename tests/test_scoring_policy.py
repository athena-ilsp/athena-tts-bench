from greekttsbench.metrics import (
    error_rates,
    fold_accents,
    normalize_for_scoring,
    summarize_error_rates,
)


def test_digit_reference_matches_verbalized_hypothesis():
    rates = error_rates(
        "Η Ελλάδα είναι μέλος του ΝΑΤΟ από το 1952.",
        "Η Ελλάδα είναι μέλος του ΝΑΤΟ από το χίλια εννιακόσια πενήντα δύο.",
        "el",
    )
    assert rates["wer"] == 0.0
    assert rates["wer_orthographic"] > 0.0


def test_accent_variance_does_not_count_against_tts():
    rates = error_rates("Γεια σου κόσμε", "Γεια σου κοσμε", "el")
    assert rates["wer"] == 0.0
    assert rates["wer_orthographic"] > 0.0


def test_abbreviation_equivalence():
    rates = error_rates(
        "Φέρε ψωμί, τυρί κ.λπ. αύριο", "Φέρε ψωμί, τυρί και λοιπά αύριο", "el"
    )
    assert rates["wer"] == 0.0


def test_acronyms_are_not_equated_with_expansions():
    rates = error_rates(
        "Μέλος του ΟΗΕ.", "Μέλος του Οργανισμού Ηνωμένων Εθνών.", "el"
    )
    assert rates["wer"] > 0.0


def test_english_skips_greek_verbalization():
    assert normalize_for_scoring("The year 2024 was great.", "en") == (
        "the year 2024 was great"
    )


def test_real_errors_still_count():
    rates = error_rates("Γεια σου κόσμε", "Γεια σας κόσμε", "el")
    assert rates["wer"] > 0.0


def test_fold_accents_handles_final_sigma():
    assert fold_accents("κόσμος") == "κοσμοσ"


def test_summarize_error_rates_groups():
    rows = [
        {
            "tts_system_id": "sys_a",
            "preprocessing_method": "raw",
            "scenario": "numbers",
            "wer": 0.5,
            "cer": 0.2,
            "wer_orthographic": 0.6,
            "cer_orthographic": 0.3,
        },
        {
            "tts_system_id": "sys_a",
            "preprocessing_method": "regex",
            "scenario": "numbers",
            "wer": 0.1,
            "cer": 0.05,
            "wer_orthographic": 0.2,
            "cer_orthographic": 0.1,
        },
    ]
    summary = summarize_error_rates(rows)
    assert summary["overall"]["count"] == 2
    assert summary["overall"]["mean_wer"] == 0.3
    assert summary["by_method"]["regex"]["mean_wer"] == 0.1
    assert summary["by_scenario_method"]["numbers::raw"]["mean_wer"] == 0.5
    assert summary["by_system_method"]["sys_a::regex"]["mean_cer"] == 0.05
