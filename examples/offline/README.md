# Offline fixture

Three newly authored Greek sentences exercise dataset validation, raw/rule
normalization, and normalization-aware scoring. `example.org` is placeholder
provenance; this fixture contains no paper text, audio, or participant record.

Run `python scripts/offline_example.py` from the repository root, after
installing the package. It checks raw-text preservation, rule expansion, and
a fixed policy-WER/orthographic-WER contrast, then prints the computed scores.
