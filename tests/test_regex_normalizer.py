import json
from pathlib import Path

from preprocessing.regex_normalizer import GreekRegexNormalizer, num_to_greek


def test_regex_normalizer_handles_large_currency_amounts():
    normalizer = GreekRegexNormalizer()
    result = normalizer.normalize("Ο προϋπολογισμός ανήλθε στα 467,590 δισ. €.")
    assert (
        "τετρακόσια εξήντα επτά δισεκατομμύρια "
        "πεντακόσια ενενήντα εκατομμύρια ευρώ"
    ) in result


def test_regex_normalizer_handles_common_ordinals():
    normalizer = GreekRegexNormalizer()
    result = normalizer.normalize("Κατέχει την 95η θέση.")
    assert "ενενηκοστή πέμπτη θέση" in result


def test_regex_normalizer_expands_million_scale_numbers():
    normalizer = GreekRegexNormalizer()
    result = normalizer.normalize("Ο πληθυσμός είναι 10.482.487 κάτοικοι.")
    assert "εκατομμύρια" in result
    assert "10.482.487" not in result


def test_regex_normalizer_processes_all_dataset_sentences():
    normalizer = GreekRegexNormalizer()
    fixture_dir = Path(__file__).resolve().parents[1] / "examples/offline/datasets"
    paths = sorted(fixture_dir.glob("*.json"))
    assert paths, "Offline fixture must be present"
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for sentence in data["sentences"]:
            normalizer.normalize(sentence["text"])


def test_thousands_group_agrees_with_feminine_chiliades():
    # "χιλιάδες" is feminine: 482 thousand must be "τετρακόσιες ογδόντα δύο".
    assert "τετρακόσιες ογδόντα δύο χιλιάδες" in num_to_greek(10_482_487)
    assert "τέσσερις χιλιάδες" in num_to_greek(4_000)


def test_no_spurious_kai_in_numbers():
    assert num_to_greek(2024) == "δύο χιλιάδες είκοσι τέσσερα"
    assert num_to_greek(1005) == "χίλια πέντε"


def test_feminine_units_and_hundreds():
    assert num_to_greek(3, "feminine") == "τρεις"
    assert num_to_greek(1, "feminine") == "μία"
    assert num_to_greek(1500, "feminine") == "χίλιες πεντακόσιες"
    assert num_to_greek(13, "feminine") == "δεκατρείς"


def test_hundred_uses_ekaton_before_continuation():
    assert num_to_greek(101) == "εκατόν ένα"
    assert num_to_greek(100) == "εκατό"


def test_day_of_month_is_feminine():
    normalizer = GreekRegexNormalizer()
    assert "τρεις Φεβρουαρίου" in normalizer.normalize(
        "Η συνάντηση είναι στις 3 Φεβρουαρίου."
    )
    assert "πρώτη Ιανουαρίου" in normalizer.normalize(
        "Η πρωτοχρονιά είναι την 1η Ιανουαρίου."
    )
    assert "τρεις Φεβρουαρίου χίλια οκτακόσια τριάντα" in normalizer.normalize(
        "Αναγνωρίστηκε στις 3 Φεβρουαρίου 1830."
    )


def test_ordinal_teens_agree_in_gender():
    normalizer = GreekRegexNormalizer()
    assert "δέκατη τρίτη θέση" in normalizer.normalize("Κατέχει την 13η θέση.")


def test_phone_numbers_read_digit_by_digit():
    normalizer = GreekRegexNormalizer()
    result = normalizer.normalize("Καλέστε στο +30 210 1234567.")
    assert "ένα δύο τρία τέσσερα πέντε έξι επτά" in result
    assert "εκατομμύριο" not in result


def test_decimals_negatives_and_fractions():
    normalizer = GreekRegexNormalizer()
    assert "μείον πέντε κόμμα τρία" in normalizer.normalize(
        "Η θερμοκρασία έπεσε στους -5,3°C."
    )
    assert "τρία τέταρτα" in normalizer.normalize("Τα 3/4 του πληθυσμού.")
    assert "δώδεκα κόμμα πέντε" in normalizer.normalize("Το ποσοστό ήταν 12,5.")


def test_scaled_currency_decimal_counts_thousandths():
    normalizer = GreekRegexNormalizer()
    # "467,5 δισ." is 467 billion 500 million, not 5 million.
    result = normalizer.normalize("Ανήλθε στα 467,5 δισ. €.")
    assert "πεντακόσια εκατομμύρια" in result
    result = normalizer.normalize("Κόστισε 3,5 εκ. €.")
    assert "πεντακόσιες χιλιάδες" in result

