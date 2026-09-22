"""Synthetic API/HTML fixtures: tests never contact Wikipedia."""

from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import pytest

from greekttsbench.datasets import validate_dataset_file
from greekttsbench.wikipedia import (
    VERSION, WikipediaClient, WikipediaError, build_dataset, collect_snapshot, digest,
)
import greekttsbench.wikipedia as wiki


def snapshot(html="<p>A newly written sentence for this synthetic test.</p>", language="en"):
    return {
        "schema": VERSION, "edition": language, "language": language,
        "api_endpoint": f"https://{language}.wikipedia.org/w/api.php",
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "rights_info": {"text": "Synthetic fixture; not Wikipedia content", "url": "https://example.org"},
        "selection": {"titles": ["Fixture"], "categories": [], "max_pages": 1},
        "skipped": [], "pages": [{
            "requested_title": "Fixture", "title": "Fixture", "page_id": 1,
            "revision_id": 42, "revision_timestamp": "2026-01-01T00:00:00Z",
            "article_url": "https://example.org/Fixture",
            "revision_url": "https://example.org/?oldid=42",
            "history_url": "https://example.org/?action=history",
            "html": html, "html_sha256": digest(html),
        }],
    }


def build(html, **kwargs):
    return build_dataset(snapshot(html), dataset_id="fixture_v1", **kwargs)


def test_html_cleanup_rejects_semantic_markup_and_preserves_prose(tmp_path):
    html = '''<div class="hatnote"><p>Navigation text should disappear.</p></div>
      <table><tr><td><p>A table cell should disappear.</p></td></tr></table>
      <p>A <b>new</b> sentence &amp; <a>linked phrase</a><sup class="reference">[1]</sup> survives.</p>
      <p>A formula <math>x = 2</math> must not become broken prose.</p>
      <p>The chemical H<sub>2</sub>O should be reviewed separately.</p>
      <p>Code <code>print(1)</code> should be reviewed separately.</p>
      <p>A sentence containing an <img src="fake"/> image is rejected.</p>
      <ul><li><p>A list paragraph should disappear.</p></li></ul>
      <p>Another sentence<br/>continues on the next line.</p>'''
    data = build(html)
    assert {row["text"] for row in data["sentences"]} == {
        "A new sentence & linked phrase survives.", "Another sentence continues on the next line.",
    }
    assert data["collection"]["counts"]["markup_paragraphs_rejected"] == 4
    assert all(row["review_status"] == "needs_language_review" for row in data["sentences"])
    assert data["sentences"][0]["source_url"].endswith("oldid=42")
    output = tmp_path / "dataset.json"
    output.write_text(json.dumps(data, ensure_ascii=False))
    assert validate_dataset_file(output, strict_word_count=True, word_count_tolerance=0) == []


@pytest.mark.parametrize(("language", "text", "expected"), [
    ("el", "Μια μικρή δοκιμή. Άλλη μία πρόταση!", ["Μια μικρή δοκιμή.", "Άλλη μία πρόταση!"]),
    ("zh", "這是一個測試。這是第二句！", ["這是一個測試。", "這是第二句！"]),
    ("hi", "यह एक परीक्षण है। यह दूसरा वाक्य है।", ["यह एक परीक्षण है।", "यह दूसरा वाक्य है।"]),
    ("ar", "هذا نص تجريبي. هذه جملة أخرى!", ["هذا نص تجريبي.", "هذه جملة أخرى!"]),
])
def test_unicode_text_is_preserved(language, text, expected):
    data = build_dataset(snapshot(f"<p>{text}</p>", language), dataset_id="unicode", min_chars=1)
    assert sorted(row["text"] for row in data["sentences"]) == sorted(expected)
    assert data["language"] == language
    assert data["collection"]["settings"]["min_words"] == 0


def test_custom_boundary_and_paragraph_mode():
    html = "<p>Πρώτη ερώτηση; Δεύτερη πρόταση.</p>"
    data = build(html, min_chars=1, boundary_pattern=r"(?<=[.;])\s+")
    assert len(data["sentences"]) == 2
    data = build(html, split_mode="paragraph")
    assert data["sentences"][0]["text"] == "Πρώτη ερώτηση; Δεύτερη πρόταση."


def test_filters_deduplication_and_audit_counts():
    html = """<p>We keep 12 small green items.</p><p>We keep 12 small green items.</p>
      <p>We keep 13 blue items.</p><p>A sentence without a numeral.</p><p>Short.</p>"""
    data = build(html, min_chars=10, include_pattern=r"\d", exclude_pattern="blue")
    assert [row["text"] for row in data["sentences"]] == ["We keep 12 small green items."]
    counts = data["collection"]["counts"]
    assert counts["duplicate_text"] == counts["include_pattern"] == counts["exclude_pattern"] == 1
    assert counts["character_length"] == 1
    assert data["sentences"][0]["characteristics"] == {"has_digit": True}
    assert build(html, min_chars=1, max_words=1)["total_sentences"] == 1


def test_sampling_is_repeatable_and_ids_survive_changed_limits():
    html = "".join(f"<p>Example number {i} is newly written for testing.</p>" for i in range(30))
    all_rows = build(html)["sentences"]
    a = build(html, max_sentences=5, seed=42)
    assert a == build(html, max_sentences=5, seed=42)
    assert a["sentences"] != build(html, max_sentences=5, seed=43)["sentences"]
    assert {row["id"] for row in a["sentences"]} <= {row["id"] for row in all_rows}
    assert len(set(row["id"] for row in all_rows)) == 30


def test_modified_snapshot_is_rejected():
    data = snapshot()
    data["pages"][0]["html"] += "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        build_dataset(data, dataset_id="bad")


@pytest.mark.parametrize("options", [
    {"min_chars": 0}, {"min_chars": 40, "max_chars": 20}, {"max_sentences": 0},
    {"min_words": 5, "max_words": 2}, {"boundary_pattern": r"(\s+)"},
    {"boundary_pattern": ""}, {"scenario": ""},
])
def test_invalid_build_options(options):
    with pytest.raises(ValueError):
        build_dataset(snapshot(), dataset_id="bad", **options)


class FakeAPI:
    language = "en"
    base_url = "https://en.wikipedia.org"
    endpoint = base_url + "/w/api.php"

    def __init__(self, mismatch=False):
        self.calls = []
        self.mismatch = mismatch

    def category_titles(self, category, limit):
        return ["Alias", "Missing", "Disambiguation", "Talk:Fixture"][:limit]

    def request(self, **params):
        self.calls.append(params)
        if "meta" in params:
            return {"query": {"general": {"lang": "en"}, "rightsinfo": {"text": "fixture"}}}
        if params["action"] == "parse":
            return {"parse": {"revid": 99 if self.mismatch else 42, "pageid": 1,
                              "text": "<p>A newly written sentence for testing.</p>"}}
        title = params["titles"]
        page = {"pageid": 1, "title": "Fixture", "ns": 0,
                "revisions": [{"revid": 42, "timestamp": "2026-01-01T00:00:00Z"}]}
        if title == "Missing":
            page = {"title": title, "missing": True}
        elif title == "Disambiguation":
            page.update(pageid=2, pageprops={"disambiguation": ""})
        elif title.startswith("Talk:"):
            page["ns"] = 1
        return {"query": {"pages": [page]}}


def test_collection_fixes_revision_and_reports_skipped_pages():
    api = FakeAPI()
    result = collect_snapshot(api, ["Fixture"], ["Category:Fixture"], 10)
    assert len(result["pages"]) == 1
    assert {row["reason"] for row in result["skipped"]} == {
        "duplicate_resolved_page", "missing_or_invalid", "disambiguation", "non_article_namespace",
    }
    assert next(call for call in api.calls if call["action"] == "parse")["oldid"] == 42
    assert result["pages"][0]["html_sha256"] == digest(result["pages"][0]["html"])
    assert len(collect_snapshot(FakeAPI(), ["Fixture", "Missing"], [], 1)["pages"]) == 1
    with pytest.raises(WikipediaError, match="mismatch"):
        collect_snapshot(FakeAPI(mismatch=True), ["Fixture"], [])


def test_category_continuation_is_bounded(monkeypatch):
    client = WikipediaClient("en", "Fixture/1.0 (https://example.org)")
    calls = []

    def request(**params):
        calls.append(params)
        if len(calls) == 1:
            return {"query": {"categorymembers": [{"title": "A"}]},
                    "continue": {"continue": "-||", "cmcontinue": "next"}}
        return {"query": {"categorymembers": [{"title": "B"}]}}

    monkeypatch.setattr(client, "request", request)
    assert client.category_titles("Category:Fixture", 2) == ["A", "B"]
    assert calls[1]["cmcontinue"] == "next"
    assert calls[1]["cmlimit"] == 1
    assert calls[0]["cmnamespace"] == 0


def stub_network(monkeypatch, replies):
    calls, sleeps = [], []
    queue = iter(replies)

    def open_url(request, timeout):
        calls.append(request)
        result = next(queue)
        if isinstance(result, Exception):
            raise result
        return BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(wiki, "urlopen", open_url)
    monkeypatch.setattr(wiki.time, "sleep", sleeps.append)
    return calls, sleeps


def test_api_retries_maxlag_and_identifies_client(monkeypatch):
    calls, sleeps = stub_network(monkeypatch, [{"error": {"code": "maxlag"}}, {"query": {}}])
    client = WikipediaClient("el", "Fixture/1.0 (https://example.org)")
    assert client.request(action="query") == {"query": {}}
    assert len(calls) == 2 and 5 in sleeps
    assert calls[0].get_header("User-agent").startswith("Fixture/")
    assert parse_qs(urlsplit(calls[0].full_url).query)["maxlag"] == ["5"]


def test_api_retry_after_and_permanent_errors(monkeypatch):
    error = HTTPError("https://example.org", 429, "limited", {"Retry-After": "7"}, None)
    _, sleeps = stub_network(monkeypatch, [error, {"query": {}}])
    WikipediaClient("en", "fixture").request(action="query")
    assert 7 in sleeps
    calls, _ = stub_network(monkeypatch, [{"error": {"code": "badvalue"}}])
    with pytest.raises(WikipediaError, match="API error"):
        WikipediaClient("en", "fixture").request(action="query")
    assert len(calls) == 1
    calls, _ = stub_network(monkeypatch, [URLError("offline")] * 4)
    with pytest.raises(WikipediaError, match="retry later"):
        WikipediaClient("en", "fixture").request(action="query")
    assert len(calls) == 4


def test_cli_offline_roundtrip_and_no_overwrite(tmp_path):
    source = tmp_path / "snapshot.json"
    source.write_text(json.dumps(snapshot()))
    output = tmp_path / "datasets" / "candidates.json"
    script = Path(__file__).resolve().parents[1] / "scripts/collect_wikipedia.py"
    command = [sys.executable, str(script), "build", "--snapshot", str(source),
               "--dataset-id", "test_candidates", "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    content = output.read_bytes()
    assert json.loads(content)["collection"]["snapshot_sha256"]
    assert validate_dataset_file(output, strict_word_count=True) == []
    assert subprocess.run(command, capture_output=True).returncode == 1
    assert output.read_bytes() == content
    command[-1] = str(tmp_path / "empty.json")
    result = subprocess.run(command + ["--min-chars", "299"], capture_output=True, text=True)
    assert result.returncode == 1 and "No candidates" in result.stderr
    assert not (tmp_path / "empty.json").exists()
