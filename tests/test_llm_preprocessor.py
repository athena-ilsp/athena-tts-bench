from preprocessing.llm_preprocessor import (
    LLMPreprocessor,
    clean_llm_output,
    strip_parenthetical_acronyms,
    validate_llm_output,
)


def test_clean_llm_output_strips_code_fences():
    assert clean_llm_output("```text\nΓεια σου κόσμε\n```") == "Γεια σου κόσμε"


def test_clean_llm_output_strips_wrapping_quotes_and_labels():
    assert clean_llm_output("«Γεια σου κόσμε»") == "Γεια σου κόσμε"
    assert clean_llm_output("Έξοδος: Γεια σου κόσμε") == "Γεια σου κόσμε"


def test_clean_llm_output_keeps_internal_quotes():
    text = "Ο πρωθυπουργός δήλωσε: «Η οικονομία ανακάμπτει» χθες."
    assert clean_llm_output(text) == text


def test_validate_llm_output_flags_empty_refusal_and_length():
    assert validate_llm_output("κείμενο", "") == ["empty output"]
    assert any(
        "refusal" in w
        for w in validate_llm_output("κείμενο εδώ", "Λυπάμαι, δεν μπορώ να βοηθήσω")
    )
    assert any("short" in w for w in validate_llm_output("α" * 100, "β" * 10))
    assert any("long" in w for w in validate_llm_output("αβ", "γ" * 100))


def test_validate_llm_output_accepts_normal_expansion():
    original = "Το ΑΕΠ αυξήθηκε κατά 3,2% το 2023."
    expanded = (
        "Το Ακαθάριστο Εγχώριο Προϊόν αυξήθηκε κατά τρία κόμμα δύο τοις εκατό "
        "το δύο χιλιάδες είκοσι τρία."
    )
    assert validate_llm_output(original, expanded) == []


def test_strip_parenthetical_acronyms_removes_restated_acronym():
    assert (
        strip_parenthetical_acronyms("το Διεθνές Νομισματικό Ταμείο (ΔΝΤ) ανακοίνωσε")
        == "το Διεθνές Νομισματικό Ταμείο ανακοίνωσε"
    )
    assert (
        strip_parenthetical_acronyms("η Ευρωπαϊκή Κεντρική Τράπεζα (ΕΚΤ).")
        == "η Ευρωπαϊκή Κεντρική Τράπεζα."
    )
    # dotted acronym form
    assert strip_parenthetical_acronyms("μια εταιρεία (Α.Ε.)") == "μια εταιρεία"


def test_strip_parenthetical_acronyms_keeps_real_parentheticals():
    # lowercase / word parentheticals are legitimate for speech and must stay
    text = "η Ελληνική Στατιστική Αρχή (απογραφή δύο χιλιάδες είκοσι ένα)"
    assert strip_parenthetical_acronyms(text) == text
    phonetic = "Το software (σόφτγουερ) εγκαταστάθηκε"
    assert strip_parenthetical_acronyms(phonetic) == phonetic


def test_normalize_with_report_strips_parenthetical_acronym():
    stub = _StubPreprocessor("το Διεθνές Νομισματικό Ταμείο (ΔΝΤ) και η ΕΚΤ συνεργάζονται")
    normalized, warnings = stub.normalize_with_report("το ΔΝΤ και η ΕΚΤ συνεργάζονται")
    assert "(ΔΝΤ)" not in normalized
    assert normalized == "το Διεθνές Νομισματικό Ταμείο και η ΕΚΤ συνεργάζονται"
    assert warnings == []


class _StubPreprocessor(LLMPreprocessor):
    """LLMPreprocessor with a canned raw response and no client/model."""

    def __init__(self, response: str):
        self._response = response
        self.provider = "stub"
        self.model = "stub"

    def _generate_raw(self, text: str, prompt: str) -> str:
        return self._response


def test_normalize_with_report_cleans_good_output():
    stub = _StubPreprocessor("«Το κείμενο κανονικοποιήθηκε σωστά εδώ»")
    normalized, warnings = stub.normalize_with_report("Το κείμενο για κανονικοποίηση εδώ")
    assert normalized == "Το κείμενο κανονικοποιήθηκε σωστά εδώ"
    assert warnings == []


def test_normalize_with_report_falls_back_on_refusal():
    original = "Το κείμενο για κανονικοποίηση"
    stub = _StubPreprocessor("Λυπάμαι, δεν μπορώ να επεξεργαστώ αυτό το αίτημα.")
    normalized, warnings = stub.normalize_with_report(original)
    assert normalized == original
    assert warnings
