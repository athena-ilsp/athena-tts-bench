# Environments

The offline code has no runtime dependencies beyond Python 3.10+.
Tests require `pytest>=8`; install with `python -m pip install -e '.[dev]'`.
CI exercises Python 3.10 and 3.12 and builds/installs the package.

Optional extras in `pyproject.toml` describe integration dependencies inherited
from the development implementation. They are not historical paper environment
locks or a guarantee that every latest upstream version is compatible.
UTMOS and the local normalizer now have explicit extras as well.

Executed ASR, UTMOS, ECAPA, Krikri, Gemini, and synthesis environments will be
recorded separately with versions/revisions supported by original run evidence.
Unknown historical versions stay marked unknown. The basic installation and
CI do not install these integrations or contact hosted models.

## Initial local validation

The first release step was built and installed in a fresh virtual environment
with Python 3.12.3 and pytest 9.1.1. All 76 unit tests passed; the installed
packages import outside the source tree, `pip check` reports no broken
requirements, and the offline fixture validates without warnings. The example
produces policy WER/CER 0 and orthographic WER 0.8. Raw/rule preprocessing
also completed. The 19 imported implementation files match their SHA-256
manifest. Python 3.10 is configured in CI but has not been run locally.
GitHub Actions will first run after the maintainer pushes.

## Wikipedia collector validation (2026-09-22)

The suite now has 97 passing tests on Python 3.12.3, including 21 new cases for
Unicode extraction, provenance, deterministic sampling, filters, category
continuation, API retries, offline CLI replay, and refusal to overwrite data.
All API unit tests use synthetic responses and require no network access.

Separate live smoke checks fetched one article from each of English, Greek,
and Chinese Wikipedia and built 10 candidate records per edition. All three
datasets passed structural validation with exact rough-word-count matching.
Greek and Chinese offline rebuilds were byte-identical. These are integration
checks, not evidence of equivalent linguistic quality across languages.
Downloaded smoke-test snapshots and candidates are not included in Git.
