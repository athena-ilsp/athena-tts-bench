# Code provenance

The code foundation was curated from the evaluation working tree based on
commit `74191335e9bd4d9d7ef5b7475912a8498947b0ef`.
The working tree also contains uncommitted paper/study work, which is kept in
a private baseline. No private development Git history is imported.

`imported_files.sha256` records the original bytes of the included Python
packages and six existing command-line scripts. From the repository root:

```bash
sha256sum -c docs/imported_files.sha256
```

All listed files retain their original content and paths. The initial public
commit changes packaging metadata, adds public documentation and an offline
example, and adapts two existing tests to use explicitly synthetic fixtures.
Those tests assert that fixtures exist, preventing an empty dataset directory
from producing a misleading pass. Other imported tests are unchanged.

The offline fixture is a newly authored three-sentence example with
`example.org` placeholder provenance. It is not a sampled paper input.
Paper data, analysis scripts, real prompts/configuration, and results will be
curated in subsequent steps. Embedded generic prompt templates already in the
imported modules retain their original bytes; their presence alone does not
identify the executed paper settings.

The frozen paper-input export is now included separately. Its
[`source_inventory.json`](../releases/slt2026/source_inventory.json) records
the 73 private-source file hashes used for extraction and verification;
[`inputs.sha256`](../releases/slt2026/inputs.sha256) pins the exported inputs.
See the [paper input guide](SLT2026_INPUTS.md) for transformations, historical
metadata differences, and the derived held-out transcription jobs.
