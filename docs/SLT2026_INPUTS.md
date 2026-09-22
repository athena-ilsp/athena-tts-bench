# Frozen SLT 2026 paper inputs

This export contains the saved inputs to the accepted experiment. It selects
existing sentence/utterance IDs and copies recorded text; it does not recollect
Wikipedia, resample sentences, renormalize text, synthesize audio, or rescore.
It is separate from the optional new-language collection workflow.

## Verify the export

From the repository root, with Python 3.10+:

```bash
python scripts/validate_slt2026_inputs.py
sha256sum -c releases/slt2026/inputs.sha256
```

No models, GPU, credentials, downloads, or private source files are needed.
Validation checks hashes, the full sentence/condition/system product, text
joins, system counts, reference mappings, and VITS voice balance. Tests include
intentional text, speaker, reference, count, and hash corruption.

| Material | Location | Count |
| --- | --- | --- |
| Original selected texts and source metadata | `datasets/slt2026/source/` | 140; 20 in each of 7 scenarios |
| Frozen raw/rule/LLM variants | `datasets/slt2026/normalized/{raw,regex,llm}/` | 420 sentence-condition inputs |
| Paper protocol and exact sentence IDs | `configs/slt2026/paper_protocol.json` | 12 system slots |
| System, model, and reference registries | `configs/slt2026/` | 12 slots, 8 model assets, 83 reference assets |
| Core synthesis, transcription, speaker-reference manifests | `manifests/slt2026/*_core.jsonl` | 5,040 rows per manifest |
| Held-out synthesis, transcription, speaker-reference manifests | `manifests/slt2026/*_heldout.jsonl` | 560 rows per manifest |
| Private-source file hashes and export audit | `releases/slt2026/` | Provenance and consistency checks |

The public default paper input set is the saved `slt2026` preset above. The
larger development pool and excluded `vits_el_1gpu` system are not exported.
Raw IDs remain unchanged; paper display labels are additional fields.
The held-out track has 80 sentence IDs (40 per natural-speaker gender), routed
across the appropriate systems. Its 560 clips are not the deployed/rated
human-study subset; trial mapping and sanitized ratings follow separately.

## Text and historical metadata

Source text, URLs, characteristics, normalized text, and per-sentence warning
records are preserved from the original artifacts. Source URLs do not always
mean verbatim Wikipedia extraction: original dataset descriptions also name
curated/synthetic material. Article revisions and per-sentence copied/adapted
status were not recorded; this export does not invent them or replace historical
text with current Wikipedia text. Dataset terms/attribution still need final
owner review before a completed release license is asserted.

The original `word_count` metadata differs from the existing rough tokenizer
for 123 selected sentences, with 33 differences exceeding the validator's
usual tolerance. These metadata values are kept for provenance; recorded
manifest token counts are also preserved. `input_export_audit.json` lists the
differences. The generic dataset validator therefore reports 33 warnings for
the source directory; strict word-count mode intentionally fails on these
historical records. The release-specific validator checks the frozen export
without changing the experiment to eliminate those warnings.

Nine selected LLM-normalized inputs carry saved warning flags; the frozen
outputs, including fallback behavior and any residual model commentary, are
kept exactly. Four sentence-condition inputs had whitespace collapsed for
line-oriented synthesis handoffs, affecting 48 core clips. Each synthesis row
therefore has both `synthesis_text` (the original normalized/scoring input) and
`handoff_text` (the exact saved handoff input). This whitespace transformation
is verified against the historical handoff maps. Do not clean or regenerate
these texts when reproducing the accepted experiment.

Original source and normalization metadata is retained except: datasets are
filtered to the selected IDs and their totals become 20; the private Krikri
path becomes its recorded public model ID; selected warning totals are
recomputed while original pool totals remain separately recorded. JSON
serialization is normalized. None of these changes alter sentence text.

## Voices, references, and generation settings

The VITS slot uses both speakers: 0=female and 1=male. The core has 210 clips
per speaker, balanced 10/10 sentences in each scenario with the same assignment
across normalization methods. Held-out VITS has 40 clips per speaker. Saved
handoff maps establish these assignments; the obsolete generic “speaker 0”
metadata was replaced with the correct per-row value.

Core speaker similarity uses shared female/male enrollment references except
for the independent Colab Parler model, which uses its own male enrollment.
Held-out speaker similarity uses same-sentence natural anchors for the other
systems and the separate enrollment for Colab. Original reference-kind and
`gender_match` flags are preserved. References were checked against the actual
saved speaker-similarity score records, not just generic configuration.

Parler's four det/LLM-caption slots retain the recorded description strings and
sampled decoding (`do_sample=True`, temperature 1.0). The saved driver/wrappers
supply seed 1234; missing checkpoint generation settings are explicitly unknown.
The independent Colab model has a different driver that inherits its checkpoint
settings; those must not be replaced with the four-slot Parler settings.
Some Chatterbox settings are supported only by recorded system metadata.
The registry distinguishes that evidence from confirmed driver parameters.

## Paths and what is still missing

Audio and transcript paths are relative to the repository root and retain the
historical directory layout. Private absolute enrollment paths become explicit
reference asset paths. Model locations become `asset:slt2026/<id>` identifiers;
`model_assets.json` reserves a `local_path` binding for actual checkpoints.
These logical identifiers are not download URLs or directly loadable model
paths. Asset download/resolution tooling follows in the asset-release step.
No audio, checkpoint, participant record, or private development history is
included here. References and models are marked unavailable in this checkout,
with access/redistribution review still pending.

The core transcription manifest is the original saved subset. The historical
held-out transcription job manifest was not found: its public version is
explicitly **derived**, using the saved synthesis inputs and no-alignment
WhisperX configuration. Its reference text and ASR identity match all 557 saved
held-out ASR score rows; transcript output locations are newly generated portable
locations, not a claim about historical execution paths.

Use the saved manifests above for the frozen experiment. The generic manifest
builders remain available for new experiments, but do not reproduce the
per-clip VITS/reference enrichment on their own. Likewise, generic LLM defaults
are not the executed Krikri configuration; use frozen normalized outputs for
paper reconstruction.

The source-side export check reconciled 5,040 core and 560 held-out rows with
the joined results, and all 5,600 references with saved speaker scores. The
scores themselves, statistical analysis, exact judge prompts/settings, human
study records, and final camera-ready table/figure verification arrive in the
next release step. This input validation is not full paper reproduction.

## Maintainer re-export

A maintainer with the original private workspace can recreate the export into
a fresh directory:

```bash
python scripts/export_slt2026_inputs.py \
  --source-root /path/to/private-evaluation-workspace \
  --output-root /tmp/slt2026-input-export
python scripts/validate_slt2026_inputs.py --root /tmp/slt2026-input-export
```

The exporter refuses existing release directories. On an error, investigate
the discrepancy and use a fresh output directory after resolving it; do not
edit accepted inputs to force the checks to pass. Private source paths remain
local command arguments. The public source inventory contains only relative
file names, lengths, and hashes.
