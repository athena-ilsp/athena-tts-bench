from greekttsbench.metrics import character_error_rate, word_error_rate


def test_word_error_rate_is_zero_for_equivalent_punctuation():
    assert word_error_rate("Γεια σου, κόσμε!", "Γεια σου κόσμε") == 0.0


def test_character_error_rate_detects_difference():
    assert character_error_rate("abc", "abd") == 1 / 3

