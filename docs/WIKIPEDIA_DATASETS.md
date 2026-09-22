# Building a dataset from Wikipedia

The collector provides a starting point for a new language dataset. It can
request an accessible Wikipedia language edition, preserve source provenance,
and produce candidate records in AthenaTTS-Bench's dataset format. It does not
guarantee correct sentence boundaries, language purity, balanced scenarios, or
the quality of the paper's manually curated Greek benchmark.

This is an optional workflow for new datasets. It does not reconstruct the
paper's 140 selected texts or change the frozen SLT experiment. No additional
Python dependencies or API keys are needed.

## 1. Choose sources and fetch a snapshot

Run from the repository root. Replace `YOUR_CONTACT_URL_OR_EMAIL` with a real
contact so Wikipedia operators can identify the client:

```bash
python scripts/collect_wikipedia.py fetch \
  --language el --title 'Γη' --title 'Σελήνη' \
  --max-pages 2 \
  --user-agent 'AthenaTTS-Bench/0.1 (YOUR_CONTACT_URL_OR_EMAIL)' \
  --snapshot outputs/wiki-el/snapshot.json
```

Use `--language fr --title 'Terre'`, for example, to request another edition.
The language argument is the Wikipedia host code; some edition codes differ
from the language code expected by ASR/TTS tools. The snapshot records both
the requested edition and the language reported by the site. An edition must
actually exist and permit API access.

Input options can be combined:

- Repeat `--title` or supply `--titles-file articles.txt` with one UTF-8 title
  per line. Article titles belong to the selected edition, not another language.
- Repeat `--category 'Category:Physics'`, using the full category title from
  that edition (a localized namespace prefix is also accepted by MediaWiki).
- `--max-pages` bounds the number of input titles attempted. Explicit titles
  come first. Category discovery takes direct article members in API sort-key
  order; it does not recurse into subcategories or randomly sample Wikipedia.
  Each category scan is also bounded by `--max-pages`. Overlaps, skipped
  pages, empty categories, and title limits can yield fewer usable articles.

Redirects are resolved. Missing pages, non-article namespaces,
disambiguation pages, and duplicate resolved pages are recorded as skips.
An API failure stops collection instead of silently producing an incomplete
snapshot. If no usable pages remain, the command exits unsuccessfully.
Commands refuse to replace existing output files: use a new versioned path.

The client makes serial requests with a delay, `maxlag`, and bounded backoff
for transient failures; long server-requested waits stop the run for a later
retry. This follows the [Action API etiquette](https://www.mediawiki.org/wiki/API:Etiquette).
Saved snapshots are reusable; use offline builds rather than fetching again
for each filter experiment. This tool is for bounded candidate collection,
not bulk mirroring.

## 2. Build candidates offline

```bash
python scripts/collect_wikipedia.py build \
  --snapshot outputs/wiki-el/snapshot.json \
  --dataset-id wiki_el_candidates_v1 \
  --min-chars 25 --max-chars 300 --max-sentences 200 --seed 42 \
  --output outputs/wiki-el/datasets/candidates.json

python scripts/validate_datasets.py \
  --datasets-dir outputs/wiki-el/datasets --strict-word-count
```

Keep snapshots outside the dataset directory: the existing validator treats
every JSON file in that directory as a dataset.

The builder extracts paragraph prose, removes citation markers and common
navigation/table/list elements, collapses whitespace, then splits and filters.
Paragraphs containing formulas, code, inline images, or non-citation
superscripts/subscripts are conservatively rejected. It does not download or
redistribute media. Case, diacritics, digits, and script characters are preserved.
HTML extraction is heuristic: unfamiliar templates can still leak into prose,
and legitimate paragraphs can be missed or rejected.

By default, boundaries recognize `. ! ?` followed by whitespace and
`。！？।॥` with or without following whitespace. This is a fallback, not a
language-aware sentence segmenter. Abbreviations, initials, quotation marks,
ellipsis, Greek `;`, Arabic `؟`, and other conventions require review.

Options for adapting the build:

| Option | Purpose and caveat |
| --- | --- |
| `--split-mode paragraph` | Keep each extracted paragraph intact for subsequent segmentation/review. Long paragraphs still face the character filter; raise `--max-chars` as needed. |
| `--boundary-pattern REGEX` | Supply a Python split expression; use non-capturing groups. For Greek, a starting example is `'(?<=[.!?;])\s+'`. This still mishandles abbreviations and quoted endings. |
| `--min-chars`, `--max-chars` | Unicode code-point length bounds; these are not grapheme, speech-duration, or phoneme counts. |
| `--min-words`, `--max-words` | Optional rough-token bounds, disabled by default. These are not linguistic word counts. |
| `--include-pattern REGEX` | Keep candidates matching a script-specific or task-specific heuristic, e.g. `'[0-9]'` for ASCII digits. It is not a scenario classifier. |
| `--exclude-pattern REGEX` | Reject a known unwanted pattern; preserve the rule in your release notes. |
| `--scenario LABEL` | Attach a provisional dataset label, default `unreviewed`. It does not validate that the text belongs to that scenario. |

Exact duplicate text is removed after whitespace cleanup. The first retained
source supplies provenance. Case, accents, Unicode normalization forms, and
near-duplicates are not collapsed. Eligible records are sorted by stable IDs;
if a cap is needed, sampling uses the supplied seed. Every row starts with
`review_status: needs_language_review`. Only `has_digit` is inferred, using
Unicode decimal-digit detection; no acronym, code-switching, or morphology
labels are guessed.

## 3. Review and adapt for the language

Before using candidates as a benchmark, involve a fluent reviewer and record
the decisions below. The same defaults need not produce comparable difficulty
or quality across languages.

| Area | What to check and decide |
| --- | --- |
| Sentence boundaries | Check initials, abbreviations, quotations, decimals, ellipses, local punctuation, and broken fragments. Use a suitable segmenter or manually split/merge paragraphs. Preserve parent IDs and document edits. |
| Word segmentation | For scripts without regular spaces, use a language-appropriate tokenizer. The schema's `word_count` is the existing rough count for compatibility; store a linguistic count separately. Do not impose Greek token limits automatically. |
| Writing system | Check combining marks, vowel marks, diacritics, zero-width joiners, bidirectional controls, script variants, and Unicode normalization. Some distinctions carry pronunciation or meaning; define policy before changing them. |
| Numerals and dates | Cover local numeral systems, decimal/group separators, currencies, units, dates, times, grammatical agreement, and ambiguous readings. Digit detection alone does not establish a numbers/dates scenario. |
| Abbreviations and names | Distinguish expanded forms from letter-by-letter or word-like readings. Check local titles, acronyms, inflection, transliteration, and foreign personal/place names with a fluent reviewer. |
| Language and dialect | Audit mixed-language passages, borrowed words, historical forms, regional varieties, and code-switching. The site's language code does not establish the language of each sentence. |
| Scenario coverage | Define the language's relevant phenomena, label the actual sentences, set quotas and difficulty, and inspect diversity. Categories describe article topics, not TTS challenges. |
| Source bias | Wikipedia prose can overrepresent formal writing, named entities, and certain domains. Check conversational/expressive coverage separately. Category order and chosen seed articles are sampling biases. |
| Duplicates and leakage | Review near-duplicates and copied passages across pages. Separate train/dev/test sources and check known training overlap where possible; collection cannot establish training independence. |
| Extraction quality | Check template remnants, truncated prose, citation removal, math/media omissions, tables/lists, and foreign-script ruby/annotation markup. Compare suspicious candidates against the saved HTML and source revision. |
| Evaluation support | Confirm suitable TTS voices, ASR coverage, transcript policy, and human raters. Validate automatic quality/speaker/judge metrics for the language rather than assuming the Greek findings transfer. |

### Existing normalization and scoring behavior

The current rule grammar and normalization prompt are Greek-specific.
`normalize_dataset_object` passes non-Greek datasets through unchanged for
the `regex` and `llm` methods. Giving those outputs different condition names
does not create meaningful normalization comparisons. Implement and validate
language-specific normalizers before claiming such comparisons.

The generic `normalize_for_scoring` currently folds accents/combining marks
even for non-Greek text, strips punctuation, and uses whitespace tokens for
WER. That can erase meaningful distinctions or yield unsuitable word units.
Define and test a language-specific scoring policy before evaluating a new
language. These existing functions are preserved for the accepted Greek
experiment; this collector does not change their behavior.

The structural validator checks fields, IDs, counts, and rough word-count
consistency. Passing it is not linguistic validation. Keep reviewed datasets
separate from candidate outputs; maintain an accept/reject/edit ledger keyed
by original ID with reviewer decisions, scenario assignments, and reasons.

## Provenance, replay, and attribution

Collection first obtains revision metadata, then requests the rendered HTML
for that revision with [`action=parse&oldid=...`](https://www.mediawiki.org/wiki/API:Parsing_wikitext).
Snapshots store HTML, its SHA-256, retrieval time, page/revision IDs and
timestamps, article/history/revision URLs, input selection, skips, and the
site's [reported rights information](https://www.mediawiki.org/wiki/API:Siteinfo).
Candidate rows retain provenance and text hashes; build metadata records the
snapshot hash, settings, rejection counts, and extraction version.

Keep the snapshot and generated JSON together with the code commit and Python
version. Rebuilding from the same snapshot/settings in that environment is
deterministic. A later live fetch may change: article/category contents and
rendered templates can evolve. Revision IDs alone do not freeze template
rendering, which is why the actual downloaded HTML is saved and verified.

Preserve attribution to the source article and contributors via the article,
revision, and history links. Record extraction and subsequent edits. Before
sharing the dataset or snapshot, check the applicable edition/page terms and
any additional notices; the captured site-wide rights label is not a complete
per-page audit. Follow the [Wikimedia reuse terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content).
Collected text is not covered by the project's future code license. The tool
does not assert that every page has identical terms.

## Completion criteria for a new dataset

Publish a dataset card describing source selection, language/script/dialect,
segmentation, filters, scenario definitions, review decisions, dataset splits,
known limitations, attribution, and hashes. Freeze the reviewed text and IDs
before synthesis. Treat any later corrected dataset as a new version.
