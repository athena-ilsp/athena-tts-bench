# SLT 2026 release progress

Camera-ready submission is complete, as reported by the authors. The release
preserves the accepted experiment; it does not reopen manuscript edits or
change the scores, sampling, exclusions, or statistical methods.

Each step is prepared as a local commit, inspected by the maintainer, and
pushed by the maintainer before the next step begins.

| Step | Scope | Status |
| --- | --- | --- |
| 1 | Public metadata, selected portable code, install specification, offline fixtures, tests, CI | Prepared in `d48fd49` |
| 2 | Exact 140 texts, 420 normalized inputs, paper protocol, system registry, core/held-out manifests | Prepared; see [input guide](SLT2026_INPUTS.md) |
| 3 | Frozen scores, sanitized study exports, versioned prompts, expected outputs and paper reproduction | Pending |
| 4 | Verified asset index/download tools, licenses, complete guides and release audit | Pending |

## Additional requested capability: Wikipedia candidate collection

Prepared as a separate commit after the code foundation: configurable Wikipedia
edition/title/category collection, revision snapshots, offline extraction and
seeded filtering, provenance, synthetic tests, and a per-language review guide.
See [Wikipedia datasets](WIKIPEDIA_DATASETS.md). This addition is authorized
before the paper-data export; the remaining four-stage release sequence and
accepted experiment are unchanged. It creates unreviewed candidates, not a
guarantee of equivalent quality across languages.

## Baseline and invariants

A private copy with SHA-256 hashes preserves the currently available source,
normalization outputs, manifests, metric records, and the 30 top-level survey
exports. Older nested survey snapshots are excluded. The local manuscript is
an anonymous historical source, not a verified copy of the submitted
camera-ready. Confirm the submitted files before pinning final paper cells.
The complete R0 table/figure input mapping remains pending.

- Preserve original sentence and utterance IDs and normalized text.
- Export the existing subset; do not resample from the larger source pool.
- Exclude `vits_el_1gpu`: 5,040 core and 560 held-out synthetic clips.
- Preserve VITS sentence-to-speaker assignment and actual generation settings.
- Preserve missing/error rows and metric-specific denominators: expected core
  counts include 18 missing WER and 4,990 parsed judge rows; held-out includes
  3 missing WER and 556 parsed judge rows. Verify during export.
- Preserve accepted aggregation, weighting, rounding, and paired tests.
- Reconcile 30 listeners, 930 primary Part-A observations, 30 repeats, and
  240 Part-B observations before publishing sanitized trial records.
- Record differences with the manuscript for author review; never force data
  or computations to match an unexplained printed value.

## Next inspection point

The frozen input export is prepared and validated. Step 3 now exports scores,
exact judge prompts/settings, approved sanitized study records, and the existing
analysis. Confirm the final submitted source before comparing paper cells.

Step 3 will supply `scripts/reproduce_paper.py --preset slt2026`, once its
inputs and calculations are verified. That command does not exist yet.
Frozen-score reproduction will run offline; waveform rescoring and
resynthesis are separate workflows.

## Inputs still to confirm

- Exact submitted manuscript/bibliography/figure for the final expected cells.
- Code license chosen by the owners, and separate text/model/audio terms.
- Consent and redistribution status for minimized study records and natural
  recordings; the original raw participant exports remain private.
- Deployed questionnaire, trial mappings, and original training projects
  where these are absent from the evaluation workspace.

Large audio files/checkpoints will be versioned download assets with hashes.
The final paper release will be immutable; later experiments get separate
presets and result directories.

## Companion training repository

The [training repository plan](TRAINING_REPOSITORY_PLAN.md) inventories our
training/data-preparation repositories, public dependencies, five corpora, and
every evaluated model asset. It defines staged migration into a separate repo
whose URL the maintainer will provide. This is planning only; training code and
private corpora have not been imported into AthenaTTS-Bench.
