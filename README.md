# AthenaTTS-Bench

Evaluation framework for complex linguistic challenges in Modern Greek
text-to-speech, accompanying the accepted IEEE SLT 2026 paper by Georgios
Syllas, Kosmas Kritsis, Georgios Paraskevopoulos, and Vassilis Katsouros.

**Release in preparation — step 1 of 4.** This checkout contains the code
foundation, an offline example, and unit tests. The paper's benchmark data,
frozen results, human-study exports, and downloadable audio/checkpoints will
arrive in subsequent steps. Full paper reproduction is not available yet.
See [release progress](docs/RELEASE_PLAN.md).

## Paper scope

The accepted experiment uses 140 sentences (20 in each of seven scenarios),
three normalization conditions (`raw`, `regex`, `llm`), and 12 system slots:
5,040 core clips, with a separate 560-clip held-out validation set and a
30-listener study. The mixed-voice VITS slot retains its original assignment.
The development-only `vits_el_1gpu` system is excluded from the paper roster.

## Offline quickstart

Use Python 3.10 or newer, from the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/validate_datasets.py --datasets-dir examples/offline/datasets --strict-word-count
python scripts/offline_example.py
python -m pytest -q
```

Installation fetches Python packages. The example and tests then run without
network access, GPU models, or API credentials. The three example sentences
are synthetic fixtures, separate from the paper's selected 140 sentences.

Build example raw/rule-normalized variants with:

```bash
python scripts/build_preprocessed_datasets.py \
  --input-dir examples/offline/datasets \
  --output-dir outputs/offline/preprocessed --method raw --method regex
```

The distribution name is `athena-tts-bench`; existing Python imports remain
`greekttsbench` and `preprocessing`. Install into a dedicated environment.

## Collect candidates for another language

A Wikipedia collector is available for new datasets. It saves article revisions
and attribution, then builds filtered, deduplicated candidates offline with
recorded settings and seeded sampling. No extra dependencies are required.

Start with [Create a dataset in your language](docs/NEW_LANGUAGE_QUICKSTART.md)
for an end-to-end example, then use the
[full collection and language review guide](docs/WIKIPEDIA_DATASETS.md). Candidates need fluent review;
collection does not guarantee benchmark quality or multilingual normalization
and scoring support. The frozen paper dataset remains a separate release step.

## Included code

- Dataset validation, normalization, and synthesis/transcription manifest builders.
- Normalization-aware WER/CER and orthographic comparison scores.
- WAV quality checks, saved judge-output parsers, and score summaries.
- Optional model wrappers for UTMOS, ECAPA, and audio judging.

The imported implementation is preserved byte-for-byte; see
[code provenance](docs/PROVENANCE.md). Generic CLI defaults are inherited from
the development pipeline: pass explicit input paths/configuration until the
paper preset is exported. The LLM API defaults are **not** the paper's Krikri
configuration. Do not regenerate the paper inputs using generic defaults.
Optional inference dependencies and executed environments still require
separate verification; see [environments](environments/README.md).

## Citation and licensing

Paper metadata is in [CITATION.cff](CITATION.cff). Proceedings/DOI metadata will
be added once confirmed. The code license remains to be selected by the
owners; see [licensing status](docs/LICENSING.md) and
[third-party notices](THIRD_PARTY_NOTICES.md).
