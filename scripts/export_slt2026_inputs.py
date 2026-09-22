#!/usr/bin/env python3
"""Curate the frozen SLT inputs from an existing private evaluation workspace.

This maintainer-only export never resamples texts, runs models, or exports ratings.
Public users validate the saved export with validate_slt2026_inputs.py instead.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greekttsbench.datasets import rough_word_count
from greekttsbench.manifests import build_transcription_manifest_rows

EXCLUDED = 'vits_el_1gpu'
LABELS = {
    'chatterbox_el_base_female': 'cb_base_F', 'chatterbox_el_base_male': 'cb_base_M',
    'chatterbox_el_lora3_female': 'cb_lora_F', 'chatterbox_el_lora3_male': 'cb_lora_M',
    'parler_tts_el_det_vasiliki': 'par_det_F', 'parler_tts_el_det_michalis': 'par_det_M',
    'parler_tts_el_llm_vasiliki': 'par_llm_F', 'parler_tts_el_llm_michalis': 'par_llm_M',
    'greek_male_lora_colab': 'par_greek_male', 'vits_el_4gpu': 'vits_F / vits_M',
    'vits_hf_single_female': 'mms_F', 'vits_hf_single_male': 'mms_M',
}
MODELS = {
    'chatterbox_el_base_female': 'chatterbox_base', 'chatterbox_el_base_male': 'chatterbox_base',
    'chatterbox_el_lora3_female': 'chatterbox_lora3', 'chatterbox_el_lora3_male': 'chatterbox_lora3',
    'parler_tts_el_det_vasiliki': 'parler_det', 'parler_tts_el_det_michalis': 'parler_det',
    'parler_tts_el_llm_vasiliki': 'parler_llm', 'parler_tts_el_llm_michalis': 'parler_llm',
    'greek_male_lora_colab': 'parler_colab', 'vits_el_4gpu': 'vits_4gpu',
    'vits_hf_single_female': 'mms_female', 'vits_hf_single_male': 'mms_male',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def key(row):
    return row['dataset_id'], row['sentence_id'], row['preprocessing_method']


def index(rows, field='utterance_id'):
    result = {row[field]: row for row in rows}
    require(len(result) == len(rows), f'Duplicate {field}')
    return result


class Export:
    def __init__(self, source, output):
        self.source, self.output = source, output
        self.sources, self.outputs = {}, []

    def read(self, name):
        data = (self.source / name).read_bytes()
        self.sources[name] = {'sha256': sha(data), 'bytes': len(data)}
        return data

    def json(self, name):
        return json.loads(self.read(name))

    def rows(self, name):
        return [json.loads(line) for line in self.read(name).splitlines() if line.strip()]

    def write(self, name, data, jsonl=False):
        path = self.output / name
        require(not path.exists(), f'Refusing to overwrite {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        if jsonl:
            content = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in data)
        else:
            content = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
        path.write_text(content, encoding='utf-8')
        self.outputs.append(name)


def reference(path):
    if path.endswith('chatterbox-tts/data/refs/female.wav'):
        return 'shared_female', 'artifacts/speaker_refs/shared_female.wav', 'target_voice'
    if path.endswith('chatterbox-tts/data/refs/male.wav'):
        return 'shared_male', 'artifacts/speaker_refs/shared_male.wav', 'target_voice'
    if path.endswith('artifacts/speaker_refs/greek_male_3.5h_ref.wav'):
        return 'colab_male', 'artifacts/speaker_refs/greek_male_3.5h_ref.wav', 'self_target_speaker'
    p = Path(path)
    if path.startswith('user_study/natural_v1/') and p.suffix == '.wav':
        return 'natural_' + p.stem, path, 'natural_anchor'
    raise ValueError(f'Unmapped reference path: {path}')


def run(source, output):
    export = Export(source, output)
    for folder in ('datasets/slt2026', 'configs/slt2026', 'manifests/slt2026', 'releases/slt2026'):
        require(not (output / folder).exists(), f'Export destination already contains {folder}')
    core_all = export.rows('manifests/synthesis_core_eval13.jsonl')
    held_all = export.rows('manifests/user_study_synth.jsonl')
    core = [deepcopy(r) for r in core_all if r['tts_system_id'] != EXCLUDED]
    held = [deepcopy(r) for r in held_all if r['tts_system_id'] != EXCLUDED]
    require(len(core_all) == 5460 and len(core) == 5040, 'Core count differs from the accepted experiment')
    require(len(held_all) == 640 and len(held) == 560, 'Held-out count differs from the accepted experiment')
    require(set(r['tts_system_id'] for r in core) == set(LABELS), 'Paper roster mismatch')
    for rows in (core, held):
        index(rows)
    config = export.json('configs/tts_systems_core13.json')
    systems = {r['id']: r for r in config['systems'] if r['id'] != EXCLUDED}
    core_refs = index(export.rows('manifests/speaker_similarity_core_eval13.jsonl'))
    held_refs = index(export.rows('manifests/user_study_spk_sim.jsonl'))
    vits = index(export.rows('manifests/core_handoff/vits_el_4gpu_audio_map.jsonl'), 'label')
    handoff_changes = []
    for sid in sorted(LABELS):
        for track, rows, folder in [('core', core, 'core_handoff'), ('heldout', held, 'user_study_handoff')]:
            maps = index(export.rows(f'manifests/{folder}/{sid}_audio_map.jsonl'), 'label')
            chosen = [r for r in rows if r['tts_system_id'] == sid]
            require(set(maps) == {r['utterance_id'] for r in chosen}, f'{track}/{sid} handoff coverage mismatch')
            for row in chosen:
                handoff = maps[row['utterance_id']]
                require(handoff['synthesis_text'] == ' '.join(row['synthesis_text'].split()), 'Unexplained handoff text mismatch')
                row['handoff_text'] = handoff['synthesis_text']
                if row['handoff_text'] != row['synthesis_text']:
                    handoff_changes.append(row['utterance_id'])
                if sid != 'vits_el_4gpu':
                    require(handoff['voice'] == row['metadata']['voice'], 'Handoff voice mismatch')
                elif track == 'heldout':
                    require(handoff['speaker'] == row['speaker'], 'Held-out VITS handoff mismatch')
    selected = defaultdict(set)
    variants = {}
    for row in core:
        selected[row['dataset_file']].add(row['sentence_id'])
        if key(row) in variants:
            require(variants[key(row)]['synthesis_text'] == row['synthesis_text'], 'System-dependent normalization')
            require(variants[key(row)]['source_text'] == row['source_text'], 'System-dependent source text')
        variants[key(row)] = row
    require(sum(map(len, selected.values())) == 140 and len(variants) == 420, 'Sentence/condition count mismatch')
    require(len(selected) == 7 and all(len(ids) == 20 for ids in selected.values()), 'Scenario selection mismatch')
    word_count_differences, warnings, sentence_index = [], [], []
    for filename, ids in sorted(selected.items()):
        original = export.json(f'datasets/{filename}')
        data = deepcopy(original)
        data['sentences'] = [s for s in original['sentences'] if s['id'] in ids]
        require(len(data['sentences']) == 20, 'Selected source sentences missing or duplicated')
        data['total_sentences'] = 20
        for s in data['sentences']:
            require(s['text'] == variants[(data['dataset_id'], s['id'], 'raw')]['source_text'], 'Source text mismatch')
            sentence_index.append({'dataset_id': data['dataset_id'], 'dataset_file': filename,
                                   'sentence_id': s['id'], 'scenario': data['scenario']})
            actual = rough_word_count(s['text'])
            if actual != s['word_count']:
                word_count_differences.append({'sentence_id': s['id'], 'recorded': s['word_count'], 'rough': actual})
        export.write(f'datasets/slt2026/source/{filename}', data)
        for method in ('raw', 'regex', 'llm'):
            normalized = deepcopy(export.json(f'artifacts/preprocessed/{method}/{filename}'))
            normalized['sentences'] = [s for s in normalized['sentences'] if s['id'] in ids]
            normalized['total_sentences'] = 20
            require(len(normalized['sentences']) == 20, 'Normalized subset missing')
            for s in normalized['sentences']:
                row = variants[(data['dataset_id'], s['id'], method)]
                require(s['normalized_text'] == row['synthesis_text'] and s['original_text'] == row['source_text'], 'Frozen normalized text mismatch')
                if s.get('normalization_warnings'):
                    warnings.append({'sentence_id': s['id'], 'method': method, 'warnings': s['normalization_warnings']})
            meta = normalized['preprocessing']
            if method == 'llm':
                meta['model'] = 'ilsp/Llama-Krikri-8B-Instruct'
                meta['model_revision'] = None
                meta['source_pool_sentences_with_warnings'] = meta['sentences_with_warnings']
                meta['sentences_with_warnings'] = sum(bool(s.get('normalization_warnings')) for s in normalized['sentences'])
            export.write(f'datasets/slt2026/normalized/{method}/{filename}', normalized)
    refs = {}
    for track, rows, source_refs in [('core', core, core_refs), ('heldout', held, held_refs)]:
        for row in rows:
            sid, uid = row['tts_system_id'], row['utterance_id']
            scored_ref = source_refs[uid]
            require(scored_ref['audio_path'] == row['audio_path'], 'Reference/audio join mismatch')
            rid, path, kind = reference(scored_ref['reference_audio_path'])
            refs[rid] = {'id': rid, 'path': path, 'kind': kind, 'availability': 'not_in_repository',
                         'redistribution_status': 'pending_review', 'download_url': None}
            row['reference_asset_id'], row['reference_audio_path'] = rid, path
            if track == 'core':
                row['reference_gender'] = scored_ref['reference_gender']
                if sid == 'vits_el_4gpu':
                    row['speaker'] = vits[uid]['speaker']
                    require(row['speaker'] == scored_ref['speaker'], 'Core VITS speaker mismatch')
                    row['metadata']['voice'] = f"sid {row['speaker']} ({row['reference_gender']})"
            else:
                for field in ('voice_gender', 'gender_match', 'reference_kind'):
                    row[field] = scored_ref[field]
                require(row.get('speaker') == scored_ref.get('speaker'), 'Held-out speaker mismatch')
            row['paper_label'] = LABELS[sid] if sid != 'vits_el_4gpu' else ('vits_F' if row['speaker'] == 0 else 'vits_M')
            row['metadata']['model_ref'] = 'asset:slt2026/' + MODELS[sid]
            if row['tts_family'] == 'chatterbox':
                gender = 'female' if sid.endswith('female') else 'male'
                row['metadata']['voice'] = 'artifacts/speaker_refs/shared_' + gender + '.wav'
        source_scores = index(export.rows(f'results/speaker_similarity_{"core_eval13" if track == "core" else "user_study"}.jsonl'))
        source_joined = index(export.rows(f'results/per_utterance_{"core_eval13" if track == "core" else "user_study"}.jsonl'))
        require({r['utterance_id'] for r in rows} == {uid for uid,r in source_joined.items() if r['tts_system_id'] != EXCLUDED}, 'Saved result coverage differs')
        for row in rows:
            require(reference(source_scores[row['utterance_id']]['reference_audio_path'])[0] == row['reference_asset_id'], 'Reference differs from actual score record')
        export.write(f'manifests/slt2026/synthesis_{track}.jsonl', rows, jsonl=True)
        fields = ('utterance_id', 'dataset_id', 'sentence_id', 'scenario', 'preprocessing_method',
                  'tts_system_id', 'tts_family', 'audio_path', 'reference_asset_id', 'reference_audio_path',
                  'reference_gender', 'voice_gender', 'gender_match', 'reference_kind', 'speaker')
        export.write(f'manifests/slt2026/speaker_references_{track}.jsonl',
                     [{k:r[k] for k in fields if k in r} for r in rows], jsonl=True)
    asr = export.json('configs/asr_whisperx_noalign.json')
    asr.pop('notes', None)
    transcription = [r for r in export.rows('manifests/transcription_core_eval13.jsonl') if r['tts_system_id'] != EXCLUDED]
    require(set(index(transcription)) == set(index(core)), 'Core ASR coverage mismatch')
    core_by_id = index(core)
    for r in transcription:
        require(r['expected_text'] == core_by_id[r['utterance_id']]['synthesis_text'], 'Core ASR reference mismatch')
    export.write('manifests/slt2026/transcription_core.jsonl', transcription, jsonl=True)
    # Historical held-out ASR job manifest is absent; these are explicitly derived jobs.
    held_transcription = build_transcription_manifest_rows(synthesis_rows=held, asr_config=asr,
                                                         transcript_dir='artifacts/transcripts')
    export.write('manifests/slt2026/transcription_heldout.jsonl', held_transcription, jsonl=True)
    saved_asr = [r for r in export.rows('results/asr_scores_user_study.jsonl') if r['tts_system_id'] != EXCLUDED]
    held_by_id = index(held)
    require(len(saved_asr) == 557, 'Held-out ASR denominator differs')
    for row in saved_asr:
        require(row['reference'] == held_by_id[row['utterance_id']]['synthesis_text'], 'Held-out ASR text mismatch')
        require(row['asr_id'] == asr['id'], 'Held-out ASR identity mismatch')
    export.write('configs/slt2026/asr.json', asr)
    llm = export.json('configs/llm_normalizer.json')
    llm.pop('notes', None)
    llm['model_revision'] = None
    export.write('configs/slt2026/normalizer.json', llm)
    for name in ('scripts/synth_parler.py', 'scripts/synth_colab_parler.py', 'scripts/build_user_study_synth.py',
                 'leonardo/slurm/core_synth/synth_parler.slurm', 'leonardo/slurm/user_synth/synth_parler.slurm',
                 'configs/experiment.json', 'preprocessing/regex_normalizer.py', 'preprocessing/llm_preprocessor.py'):
        export.read(name)
    system_rows = []
    for sid, meta in sorted(systems.items()):
        item = {k:meta[k] for k in ('id', 'family', 'display_name', 'language', 'sample_rate_hz')}
        item.update(paper_label=LABELS[sid], model_asset_id=MODELS[sid],
                    model_ref='asset:slt2026/' + MODELS[sid], enabled=True,
                    core_clips=sum(r['tts_system_id']==sid for r in core),
                    heldout_clips=sum(r['tts_system_id']==sid for r in held),
                    voice=next(r['metadata']['voice'] for r in core if r['tts_system_id']==sid))
        if sid == 'vits_el_4gpu':
            item['voice'] = 'Per-row speaker: 0=female, 1=male'
            item['speaker_assignment'] = {'core': {'0':210,'1':210}, 'heldout':{'0':40,'1':40},
                                          'source':'frozen handoff maps; never regenerate from generic metadata'}
        item['generation'] = {'status':'historical settings incomplete; exact resynthesis not claimed'}
        if sid.startswith('parler_tts_el_'):
            item['generation'].update(do_sample=True, temperature=1.0, seed=1234,
                evidence=['scripts/synth_parler.py','leonardo/slurm/core_synth/synth_parler.slurm','leonardo/slurm/user_synth/synth_parler.slurm'],
                seed_provenance='driver default; saved wrappers do not override',
                max_length='external config value; driver fallback 2580',
                min_new_tokens='decoder.num_codebooks + 1', cache_implementation=None,
                inherited_generation_config='not archived in this export')
        elif sid.startswith('chatterbox_'):
            item['generation'].update(exaggeration=0.7, cfg_weight=0.3, language_id='el',
                                      evidence_status='recorded system metadata; external inference environment pending')
        elif sid == 'greek_male_lora_colab':
            item['generation'].update(evidence=['scripts/synth_colab_parler.py'],
                                      decoding='inherits checkpoint generation_config; no decoding overrides in driver', seed=None)
        system_rows.append(item)
    export.write('configs/slt2026/systems.json', {'systems':system_rows})
    export.write('configs/slt2026/reference_assets.json', {'assets': [refs[k] for k in sorted(refs)]})
    export.write('configs/slt2026/model_assets.json', {'assets':[
        {'id':mid, 'model_ref':'asset:slt2026/'+mid, 'local_path':None, 'download_url':None,
         'availability':'not_in_repository', 'checkpoint_revision':None,
         'checkpoint_filename':'G_90000.pth' if mid=='vits_4gpu' else None}
        for mid in sorted(set(MODELS.values()))]})
    export.write('configs/slt2026/paper_protocol.json', {
        'preset':'slt2026', 'excluded_systems':[EXCLUDED], 'methods':['raw','regex','llm'],
        'counts':{'sentences':140,'scenarios':7,'sentences_per_scenario':20,'normalized_inputs':420,
                  'systems':12,'core_clips':5040,'heldout_clips':560,'heldout_sentences':80},
        'selection':{'seed':20260610,'max_source_tokens':25,'export_policy':'actual IDs from saved manifests, no resampling'},
        'sentences':sorted(sentence_index, key=lambda r:(r['dataset_id'],r['sentence_id'])),
        'system_ids': sorted(LABELS),
        'paths':{'source':'datasets/slt2026/source','normalized':'datasets/slt2026/normalized',
                 'systems':'configs/slt2026/systems.json','manifests':'manifests/slt2026'},
        'paths_relative_to':'repository root; external model assets require explicit local_path bindings',
        'heldout_transcription_status':'new portable jobs derived from saved synthesis inputs and recorded ASR config; historical job manifest unavailable',
        'historical_data_quality':{'word_count':'original metadata preserved; release audit lists differences',
                                   'normalization':'original text, warnings and fallbacks preserved'},
        'analysis_status':'frozen scores/statistics/human exports and final camera-ready cell verification follow separately',
    })
    audit = {
        'scope':'input export only; not paper-result reproduction',
        'checks':{'core_rows':len(core),'heldout_rows':len(held),'selected_sentences':140,'normalized_inputs':len(variants),
                  'core_result_joins':len(core),'heldout_result_joins':len(held),'speaker_reference_score_joins':len(core)+len(held),
                  'variant_text_mismatches':0, 'vits_core_speakers':dict(sorted(Counter(str(r['speaker']) for r in core if r['tts_system_id']=='vits_el_4gpu').items()))},
        'original_word_count_differences':word_count_differences,
        'normalization_warnings':warnings,
        'handoff_whitespace_changes':handoff_changes,
        'limitations':['Audio/checkpoints not included.', 'Source-text URLs are historical; revision IDs and copied/adapted status were not recorded per sentence.',
                       'Generic VITS speaker-0 metadata was replaced with saved per-clip assignment.',
                       'Held-out transcription jobs are derived, not an archived executed manifest.',
                       'Historical model/environment revisions and some generation settings remain unknown.'],
    }
    export.write('releases/slt2026/input_export_audit.json', audit)
    export.write('releases/slt2026/source_inventory.json', {'files':dict(sorted(export.sources.items()))})
    hash_path = output/'releases/slt2026/inputs.sha256'
    hash_path.write_text(''.join(f'{sha((output/name).read_bytes())}  {name}\n' for name in sorted(export.outputs)), encoding='utf-8')
    print(json.dumps(audit['checks'], indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args.source_root.resolve(), args.output_root.resolve())
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Export failed: {exc}\n')
