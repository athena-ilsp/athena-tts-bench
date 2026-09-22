# AthenaTTS-Bench

Evaluation framework for complex linguistic challenges in Modern Greek
text-to-speech, accompanying the accepted IEEE SLT 2026 paper by Georgios
Syllas, Kosmas Kritsis, Georgios Paraskevopoulos, and Vassilis Katsouros.

**Release in preparation — paper inputs are included.** This checkout contains
the code foundation, multilingual Wikipedia collection, and the frozen paper
texts, normalization outputs, system registry, and manifests. Frozen scores,
sanitized human-study exports, downloadable audio/checkpoints, and full paper
reproduction follow in subsequent steps. See [release progress](docs/RELEASE_PLAN.md).

## Paper scope

The accepted experiment uses 140 sentences (20 in each of seven scenarios),
three normalization conditions (`raw`, `regex`, `llm`), and 12 system slots:
5,040 core clips, with a separate 560-clip held-out validation set and a
30-listener study. The mixed-voice VITS slot retains its original assignment.
The development-only `vits_el_1gpu` system is excluded from the paper roster.

## Validate the paper inputs

```bash
python3 scripts/validate_slt2026_inputs.py
```

The [paper input guide](docs/SLT2026_INPUTS.md) describes the exact 140 texts,
420 normalized inputs, 5,040 core clips and 560 held-out clips, voice/reference
mappings, preserved historical warnings, and asset availability. Validation is
offline; audio and weights are not required.

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
and scoring support. The frozen paper dataset is kept separately under `datasets/slt2026/`.

## Included code

- Dataset validation, normalization, and synthesis/transcription manifest builders.
- Normalization-aware WER/CER and orthographic comparison scores.
- WAV quality checks, saved judge-output parsers, and score summaries.
- Optional model wrappers for UTMOS, ECAPA, and audio judging.

The imported implementation is preserved byte-for-byte; see
[code provenance](docs/PROVENANCE.md). Generic CLI defaults are inherited from
the development pipeline: use the saved paper manifests for the accepted
experiment and explicit input paths/configuration for new experiments. The LLM API defaults are **not** the paper's Krikri
configuration. Do not regenerate the paper inputs using generic defaults.
Optional inference dependencies and executed environments still require
separate verification; see [environments](environments/README.md).

## Citation and licensing

Paper metadata is in [CITATION.cff](CITATION.cff). Proceedings/DOI metadata will
be added once confirmed. The code license remains to be selected by the
owners; see [licensing status](docs/LICENSING.md) and
[third-party notices](THIRD_PARTY_NOTICES.md).
