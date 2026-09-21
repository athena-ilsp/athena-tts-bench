# Third-party components

This initial commit contains project evaluation/preprocessing source and
synthetic test fixtures. It does not vendor model weights or training corpora.
Optional integrations reference these external projects:

| Component | Upstream | Use |
| --- | --- | --- |
| WhisperX | https://github.com/m-bain/whisperX | ASR transcription |
| SpeechMOS / UTMOS | https://github.com/tarepan/SpeechMOS | Automatic naturalness |
| SpeechBrain | https://github.com/speechbrain/speechbrain | ECAPA speaker embeddings |
| Transformers | https://github.com/huggingface/transformers | Optional model loading |
| Qwen2.5-Omni | https://github.com/QwenLM/Qwen2.5-Omni | Exploratory judge wrapper, not the paper default |
| Google Gen AI SDK | https://github.com/googleapis/python-genai | Gemini audio judging |

Dependency, checkpoint, dataset, and hosted-service terms are separate from
project-owned code. Exact revisions, applicable notices, and redistribution
status will be recorded with each released environment or asset. This table
is an integration inventory, not a completed license/redistribution audit.
