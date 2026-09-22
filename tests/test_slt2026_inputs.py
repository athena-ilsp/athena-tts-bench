"""Release contracts exercise the real frozen inputs without models or private data."""
from pathlib import Path

import pytest

from scripts import validate_slt2026_inputs as release

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_release_hashes_and_cross_file_joins():
    report = release.validate(ROOT)
    assert report['sentences'] == 140
    assert report['normalized_inputs'] == 420
    assert report['core_clips'] == 5040
    assert report['heldout_clips'] == 560
    assert report['legacy_word_count_warnings'] == 33


@pytest.mark.parametrize('mutation, message', [
    ('drop', 'clip count'), ('duplicate', 'Duplicate utterance_id'),
    ('text', 'normalized text mismatch'), ('speaker', 'VITS'),
    ('reference', 'natural reference mismatch'),
])
def test_corrupted_manifest_fails_structural_checks(monkeypatch, mutation, message):
    original = release.read_jsonl

    def changed(path):
        rows = original(path)
        if path.name == 'synthesis_core.jsonl':
            if mutation == 'drop':
                rows.pop()
            elif mutation == 'duplicate':
                rows[-1] = rows[0]
            elif mutation == 'text':
                rows[0]['synthesis_text'] += ' changed'
                rows[0]['handoff_text'] = ' '.join(rows[0]['synthesis_text'].split())
            elif mutation == 'speaker':
                row = next(r for r in rows if r['tts_system_id']=='vits_el_4gpu')
                row['speaker'] = 1-row['speaker']
        elif path.name == 'synthesis_heldout.jsonl' and mutation == 'reference':
            row = next(r for r in rows if r['tts_system_id']!='greek_male_lora_colab')
            row['reference_asset_id'] = 'colab_male'
            row['reference_audio_path'] = 'artifacts/speaker_refs/greek_male_3.5h_ref.wav'
        return rows

    monkeypatch.setattr(release, 'read_jsonl', changed)
    with pytest.raises(ValueError, match=message):
        release.validate(ROOT, verify_hashes=False)


def test_changed_frozen_bytes_fail_hash_check(tmp_path):
    protocol=tmp_path/'configs/slt2026/paper_protocol.json'
    protocol.parent.mkdir(parents=True)
    protocol.write_bytes((ROOT/'configs/slt2026/paper_protocol.json').read_bytes())
    hashes=tmp_path/'releases/slt2026/inputs.sha256'
    hashes.parent.mkdir(parents=True)
    first=(ROOT/'releases/slt2026/inputs.sha256').read_text().splitlines()[0]
    hashes.write_text(first+'\n')
    name=first.split('  ',1)[1]
    changed=tmp_path/name
    changed.parent.mkdir(parents=True,exist_ok=True)
    changed.write_text('{}\n')
    with pytest.raises(ValueError,match='Hash mismatch'):
        release.validate(tmp_path)
