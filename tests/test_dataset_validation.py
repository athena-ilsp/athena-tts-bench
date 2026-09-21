from pathlib import Path

from greekttsbench.datasets import dataset_files, validate_dataset_dir


FIXTURES = Path(__file__).resolve().parents[1] / "examples/offline/datasets"


def test_dataset_dir_has_no_structural_errors():
    assert dataset_files(FIXTURES), "Offline fixture must be present"
    issues = validate_dataset_dir(FIXTURES)
    errors = [issue for issue in issues if issue.severity == "error"]
    assert errors == []


def test_dataset_word_counts_are_current_before_release():
    assert dataset_files(FIXTURES), "Offline fixture must be present"
    issues = validate_dataset_dir(FIXTURES, strict_word_count=True)
    errors = [issue for issue in issues if issue.severity == "error"]
    assert errors == []
