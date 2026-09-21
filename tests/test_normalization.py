from greekttsbench.normalization import normalize_dataset_object


def make_dataset(language: str) -> dict:
    return {
        "dataset_id": "test_v1",
        "scenario": "test",
        "language": language,
        "total_sentences": 1,
        "source": ["example.org"],
        "sentences": [
            {
                "id": "s_001",
                "text": "The year 2024 was great."
                if language == "en"
                else "Το 2024 ήταν σπουδαίο έτος.",
            }
        ],
    }


def test_english_dataset_passes_through_regex_unchanged():
    output = normalize_dataset_object(make_dataset("en"), "regex")
    sentence = output["sentences"][0]
    assert sentence["normalized_text"] == sentence["original_text"]
    assert "2024" in sentence["normalized_text"]
    assert output["preprocessing"]["normalizer"] == "identity_non_greek"


def test_english_dataset_passes_through_llm_without_building_a_model():
    # Must not construct any LLM client/model for non-Greek datasets.
    output = normalize_dataset_object(make_dataset("en"), "llm")
    sentence = output["sentences"][0]
    assert sentence["normalized_text"] == sentence["original_text"]


def test_greek_dataset_is_normalized_by_regex():
    output = normalize_dataset_object(make_dataset("el"), "regex")
    sentence = output["sentences"][0]
    assert "2024" not in sentence["normalized_text"]
    assert "δύο χιλιάδες είκοσι τέσσερα" in sentence["normalized_text"]


class _FakeLLM:
    provider = "fake"
    model = "fake"

    def normalize_with_report(self, text):
        return text + " κανονικοποιημένο", ["fake warning"]


def test_llm_warnings_are_recorded_on_sentences():
    output = normalize_dataset_object(
        make_dataset("el"), "llm", llm_preprocessor=_FakeLLM()
    )
    sentence = output["sentences"][0]
    assert sentence["normalization_warnings"] == ["fake warning"]
    assert output["preprocessing"]["sentences_with_warnings"] == 1
