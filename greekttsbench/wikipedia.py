"""Wikipedia candidate collection; no language-quality or scenario guarantees."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from html.parser import HTMLParser
import json
import random
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from greekttsbench.datasets import rough_word_count


VERSION = "wikipedia-candidates-v1"
BOUNDARY = r"(?<=[.!?。！？।॥])\s+|(?<=[。！？।॥])(?=\S)"


def digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


class WikipediaError(RuntimeError):
    """An API failure that should stop collection rather than omit data."""


class WikipediaClient:
    """Serial, bounded Action API reads with maxlag and retry/backoff."""

    def __init__(self, language: str, user_agent: str, *, delay: float = 0.5):
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,29}", language):
            raise ValueError("Use a Wikipedia edition code, e.g. el, en, or zh-min-nan")
        if not user_agent.strip() or any(ord(c) < 32 for c in user_agent):
            raise ValueError("Provide a descriptive User-Agent with contact information")
        if delay < 0.5:
            raise ValueError("Request delay must be at least 0.5 seconds")
        self.language = language
        self.base_url = f"https://{language}.wikipedia.org"
        self.endpoint = self.base_url + "/w/api.php"
        self.user_agent = user_agent
        self.delay = delay
        self._last_request = None

    def request(self, **params) -> dict:
        query = {**params, "format": "json", "formatversion": 2, "maxlag": 5}
        request = Request(self.endpoint + "?" + urlencode(query), headers={
            "User-Agent": self.user_agent, "Accept": "application/json",
        })
        for attempt in range(4):
            if self._last_request is not None:
                time.sleep(max(0, self.delay - (time.monotonic() - self._last_request)))
            self._last_request = time.monotonic()
            retry_after = 0.0
            try:
                with urlopen(request, timeout=30) as response:
                    data = json.load(response)
                if not isinstance(data, dict):
                    raise WikipediaError("API returned a non-object JSON response")
                if data.get("warnings"):
                    raise WikipediaError(f"API warning: {data['warnings']}")
                error = data.get("error")
                if not error:
                    return data
                if error.get("code") not in {"maxlag", "ratelimited"}:
                    raise WikipediaError(f"API error: {error}")
                reason = str(error)
                retry_after = 5.0
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise WikipediaError(f"HTTP {exc.code} from {self.endpoint}") from exc
                reason = f"HTTP {exc.code}"
                retry = exc.headers.get("Retry-After", "0") if exc.headers else "0"
                try:
                    retry_after = float(retry)
                except ValueError:
                    try:
                        retry_after = max(0, parsedate_to_datetime(retry).timestamp() - time.time())
                    except (ValueError, TypeError, OverflowError):
                        retry_after = 0
            except (URLError, TimeoutError) as exc:
                reason = str(exc)
            except (ValueError, UnicodeError) as exc:
                raise WikipediaError("API returned invalid JSON; no snapshot was saved") from exc
            if attempt == 3 or retry_after > 30:
                raise WikipediaError(f"Collection stopped ({reason}); retry later")
            time.sleep(max(2 ** (attempt + 1), retry_after))
        raise AssertionError("unreachable")

    def category_titles(self, category: str, limit: int) -> list[str]:
        """Direct article members only; no recursive category traversal."""
        titles = []
        continuation = {}
        while len(titles) < limit:
            data = self.request(
                action="query", list="categorymembers", cmtitle=category,
                cmnamespace=0, cmtype="page", cmsort="sortkey", cmdir="asc",
                cmlimit=min(500, limit - len(titles)), **continuation,
            )
            titles.extend(row["title"] for row in data["query"]["categorymembers"])
            continuation = data.get("continue", {})
            if not continuation:
                break
        return titles[:limit]


def collect_snapshot(client: WikipediaClient, titles: list[str], categories: list[str],
                     max_pages: int = 20) -> dict:
    """Freeze rendered HTML at explicitly requested revisions and its provenance."""
    if max_pages < 1:
        raise ValueError("max_pages must be positive")
    if not titles and not categories:
        raise ValueError("Provide at least one article title or category")
    if any(not title.strip() or "|" in title for title in titles + categories):
        raise ValueError("Titles/categories must be nonempty single titles, without '|'")
    siteinfo = client.request(action="query", meta="siteinfo", siprop="general|rightsinfo")["query"]
    selected = list(dict.fromkeys(titles))[:max_pages]
    for category in categories:
        if len(selected) >= max_pages:
            break
        # Bound discovery too. Overlapping categories may yield fewer unique pages.
        for title in client.category_titles(category, max_pages):
            if title not in selected and len(selected) < max_pages:
                selected.append(title)
    pages, skipped, seen = [], [], set()
    for title in selected:
        data = client.request(action="query", titles=title, redirects=1,
                              prop="info|revisions|pageprops", rvprop="ids|timestamp")
        page = data["query"]["pages"][0]
        reason = None
        if "missing" in page or "invalid" in page:
            reason = "missing_or_invalid"
        elif page.get("ns") != 0:
            reason = "non_article_namespace"
        elif "disambiguation" in page.get("pageprops", {}):
            reason = "disambiguation"
        elif page["pageid"] in seen:
            reason = "duplicate_resolved_page"
        if reason:
            skipped.append({"requested_title": title, "reason": reason})
            continue
        revisions = page.get("revisions", [])
        if not revisions or "revid" not in revisions[0]:
            raise WikipediaError(f"No visible revision for {title!r}")
        revision = revisions[0]
        parsed = client.request(action="parse", oldid=revision["revid"],
                                prop="text|revid", disableeditsection=1, disabletoc=1)["parse"]
        if parsed.get("revid") != revision["revid"] or parsed.get("pageid") != page["pageid"]:
            raise WikipediaError(f"Revision/page mismatch for {title!r}")
        html = parsed["text"]
        if not isinstance(html, str):
            raise WikipediaError("Expected formatversion=2 HTML text")
        seen.add(page["pageid"])
        pages.append({
            "requested_title": title, "title": page["title"], "page_id": page["pageid"],
            "revision_id": revision["revid"], "revision_timestamp": revision["timestamp"],
            "article_url": client.base_url + "/wiki/" + quote(page["title"].replace(" ", "_"), safe=""),
            "revision_url": client.base_url + "/w/index.php?oldid=" + str(revision["revid"]),
            "history_url": client.base_url + "/w/index.php?" + urlencode({"title": page["title"], "action": "history"}),
            "html": html, "html_sha256": digest(html),
        })
    return {
        "schema": VERSION, "edition": client.language,
        "language": siteinfo["general"].get("lang", client.language),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "api_endpoint": client.endpoint, "rights_info": siteinfo.get("rightsinfo", {}),
        "selection": {"titles": titles, "categories": categories, "max_pages": max_pages,
                      "resolved_input_titles": selected, "category_recursion": False},
        "pages": pages, "skipped": skipped,
    }


class _Paragraphs(HTMLParser):
    """Keep paragraph prose; conservatively reject formula/code/media paragraphs."""

    BLOCK_TAGS = {"table", "ul", "ol", "dl", "script", "style", "figure", "nav", "aside"}
    BLOCK_CLASSES = {"reference", "reflist", "mw-editsection", "hatnote", "navbox",
                     "infobox", "metadata", "thumb", "sidebar", "noprint", "toc"}
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
            "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.current = None
        self.invalid = False
        self.paragraphs = []
        self.rejected = 0

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        blocked = (bool(self.stack and self.stack[-1][1]) or tag in self.BLOCK_TAGS
                   or bool(classes & self.BLOCK_CLASSES))
        if self.current is not None and not (self.stack and self.stack[-1][1]):
            if tag in {"math", "code", "pre", "img", "sup", "sub"} and "reference" not in classes:
                self.invalid = True
            if tag == "br":
                self.current.append(" ")
        if tag == "p" and not blocked:
            self.current = []
            self.invalid = False
        if tag not in self.VOID:
            self.stack.append((tag, blocked))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == "p" and self.current is not None:
            text = " ".join("".join(self.current).split())
            if text and not self.invalid:
                self.paragraphs.append(text)
            elif self.invalid:
                self.rejected += 1
            self.current = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if self.current is not None and not (self.stack and self.stack[-1][1]):
            self.current.append(data)


def build_dataset(snapshot: dict, *, dataset_id: str, scenario: str = "unreviewed",
                  split_mode: str = "punctuation", boundary_pattern: str = BOUNDARY,
                  min_chars: int = 25, max_chars: int = 300, min_words: int = 0,
                  max_words: int = 0, max_sentences: int = 200, seed: int = 20260610,
                  include_pattern: str | None = None, exclude_pattern: str | None = None) -> dict:
    """Build deterministic, exact-text-deduplicated candidates from a saved snapshot."""
    if snapshot.get("schema") != VERSION:
        raise ValueError("Unsupported Wikipedia snapshot schema")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", dataset_id):
        raise ValueError("dataset_id must contain only letters, digits, underscores, or hyphens")
    if not scenario.strip() or split_mode not in {"punctuation", "paragraph"}:
        raise ValueError("Provide a nonempty scenario and a valid split mode")
    if min_chars < 1 or max_chars < min_chars or min_words < 0 or max_words < 0:
        raise ValueError("Invalid character/word bounds")
    if (max_words and max_words < min_words) or max_sentences < 1:
        raise ValueError("Invalid word bounds or sentence limit")
    boundary = re.compile(boundary_pattern)
    if boundary.groups or boundary.search(""):
        raise ValueError("Boundary patterns must be non-capturing and must not match empty input")
    include = re.compile(include_pattern) if include_pattern else None
    exclude = re.compile(exclude_pattern) if exclude_pattern else None
    counts = Counter()
    rows, seen = [], set()
    for page in snapshot["pages"]:
        if digest(page["html"]) != page["html_sha256"]:
            raise ValueError(f"Snapshot HTML hash mismatch for {page['title']!r}")
        parser = _Paragraphs()
        parser.feed(page["html"])
        parser.close()
        counts["paragraphs"] += len(parser.paragraphs)
        counts["markup_paragraphs_rejected"] += parser.rejected
        for paragraph_index, paragraph in enumerate(parser.paragraphs):
            chunks = boundary.split(paragraph) if split_mode == "punctuation" else [paragraph]
            for text in chunks:
                text = text.strip()
                if not text:
                    continue
                counts["candidates"] += 1
                words = rough_word_count(text)
                reason = None
                if not min_chars <= len(text) <= max_chars:
                    reason = "character_length"
                elif words < min_words or (max_words and words > max_words):
                    reason = "rough_word_length"
                elif include and not include.search(text):
                    reason = "include_pattern"
                elif exclude and exclude.search(text):
                    reason = "exclude_pattern"
                elif text in seen:
                    reason = "duplicate_text"
                if reason:
                    counts[reason] += 1
                    continue
                seen.add(text)
                identity = f"{snapshot['edition']}:{page['page_id']}:{page['revision_id']}:{text}"
                rows.append({
                    "id": "wiki_" + digest(identity)[:24], "text": text,
                    "source_url": page["revision_url"], "word_count": words,
                    "characteristics": {"has_digit": any(char.isdecimal() for char in text)},
                    "review_status": "needs_language_review",
                    "provenance": {key: page[key] for key in (
                        "title", "page_id", "revision_id", "revision_timestamp",
                        "article_url", "history_url", "html_sha256",
                    )} | {"paragraph_index": paragraph_index, "text_sha256": digest(text)},
                })
    rows.sort(key=lambda row: row["id"])
    counts["eligible_unique"] = len(rows)
    if len(rows) > max_sentences:
        rows = sorted(random.Random(seed).sample(rows, max_sentences), key=lambda row: row["id"])
    counts["selected"] = len(rows)
    return {
        "dataset_id": dataset_id, "scenario": scenario, "language": snapshot["language"],
        "total_sentences": len(rows), "source": [snapshot["api_endpoint"]], "sentences": rows,
        "collection": {
            "tool_version": VERSION, "edition": snapshot["edition"],
            "retrieved_at": snapshot["retrieved_at"], "rights_info": snapshot["rights_info"],
            "source_selection": snapshot["selection"], "skipped_pages": snapshot["skipped"],
            "settings": {"split_mode": split_mode, "boundary_pattern": boundary_pattern,
                         "min_chars": min_chars, "max_chars": max_chars, "min_words": min_words,
                         "max_words": max_words, "max_sentences": max_sentences, "seed": seed,
                         "include_pattern": include_pattern, "exclude_pattern": exclude_pattern},
            "counts": dict(sorted(counts.items())),
            "review_status": "needs_language_review",
            "transformations": ["paragraph HTML extraction", "reference-marker removal",
                                "whitespace collapse", "configured splitting and filtering"],
            "word_count_policy": "greekttsbench.datasets.rough_word_count; not linguistic tokenization",
        },
    }
