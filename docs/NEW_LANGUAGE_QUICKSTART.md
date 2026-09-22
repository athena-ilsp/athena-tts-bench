# Create a dataset in your language

Yes: you can already create a candidate text dataset from your language's
Wikipedia using this repository. You do not need Greek models, a GPU, an LLM,
or access to the paper's private training data. You need Python 3.10+, internet
access for collection, and a fluent reviewer before using the text as a benchmark.

## Collect, build, validate

From the repository root, this example creates a French candidate dataset.
Change the edition code and article titles for your language, and replace the
contact placeholder with your own email or contact URL. Run the commands in
order. Fetching needs internet; building and validation work offline.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .

python scripts/collect_wikipedia.py fetch \
  --language fr --title 'Terre' --title 'Lune' --max-pages 2 \
  --user-agent 'AthenaTTS-Bench/0.1 (YOUR_CONTACT_URL_OR_EMAIL)' \
  --snapshot outputs/wiki-fr/snapshot.json

python scripts/collect_wikipedia.py build \
  --snapshot outputs/wiki-fr/snapshot.json \
  --dataset-id wiki_fr_candidates_v1 \
  --min-chars 25 --max-chars 300 --max-sentences 100 --seed 42 \
  --output outputs/wiki-fr/datasets/candidates.json

python scripts/validate_datasets.py \
  --datasets-dir outputs/wiki-fr/datasets --strict-word-count
```

`candidates.json` is a usable dataset-format file containing text, IDs, rough
word counts, source revisions/links, and review status. Fewer than 100 records
is normal if the source pool or filters yield fewer candidates. Existing files
are never overwritten: choose a new output filename for another build.
`outputs/` is ignored by Git, so collection does not accidentally become part
of the frozen paper release.

Use `--titles-file` for a larger article list or `--category` for direct members
of a category in that edition. The [full guide](WIKIPEDIA_DATASETS.md) explains
selection limits, filtering, punctuation overrides, and attribution.

## Turn candidates into a reviewed dataset

1. Read each candidate with a fluent speaker. Remove broken sentences, unwanted
   markup, duplicates, unsuitable subject matter, and wrong-language passages.
2. Fix sentence segmentation where necessary. For languages with unfamiliar
   punctuation or without spaces, consider `--split-mode paragraph` during
   collection and apply a suitable segmenter afterward. Increase `--max-chars`
   if retaining long paragraphs. Do not assume rough word counts are linguistic
   word counts.
3. Define your evaluation scenarios and assign sentences to them. A Wikipedia
   category is a topic, not an evaluation scenario. The tool's default scenario
   is `unreviewed`; it does not classify acronyms, morphology, or code-switching.
4. Save reviewed versions in a separate directory. Keep source IDs/provenance,
   record exclusions and edits, and update `total_sentences` and `word_count`
   when text changes. The latter follows `greekttsbench.datasets.rough_word_count`;
   store any language-specific token count separately. Preserve parent IDs for
   split/merged records and assign unique new IDs. Change review status only
   after review, then run the validator on that reviewed directory.
5. Add a dataset card stating language/dialect/script, sources, scenarios,
   segmentation, filters, reviewer decisions, known gaps, attribution/terms,
   code version, and hashes. Freeze that version before synthesizing audio.

No language-specific normalizer is needed to collect text or create a raw-text
dataset. To build normalized experimental conditions or evaluate synthesized
audio, you must separately adapt normalization, tokenization/scoring, models,
and prompts for the language. Existing non-Greek `regex`/`llm` dataset processing
passes the text through unchanged; the existing scoring policy also needs
review before use outside Greek. See the [language review checklist](WIKIPEDIA_DATASETS.md#3-review-and-adapt-for-the-language).

Creating a dataset is supported today. Equivalent benchmark quality across
languages depends on the source selection and language-specific review.
