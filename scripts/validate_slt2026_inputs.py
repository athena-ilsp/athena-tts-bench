#!/usr/bin/env python3
"""Validate frozen SLT input hashes, text joins, roster, and voice assignments offline."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greekttsbench.datasets import validate_dataset_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def indexed(rows, field='utterance_id'):
    output = {r[field]:r for r in rows}
    require(len(output) == len(rows), f'Duplicate {field}')
    return output


def relative(path):
    p = PurePosixPath(path)
    require(not p.is_absolute() and '..' not in p.parts and not str(p).startswith('~'), f'Non-portable path: {path}')


def validate(root, *, verify_hashes=True):
    protocol = read_json(root/'configs/slt2026/paper_protocol.json')
    require(protocol['counts'] == {'sentences':140,'scenarios':7,'sentences_per_scenario':20,
        'normalized_inputs':420,'systems':12,'core_clips':5040,'heldout_clips':560,'heldout_sentences':80}, 'Protocol count drift')
    hashed_paths = set()
    for line in (root/'releases/slt2026/inputs.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        relative(name)
        require(name not in hashed_paths, f'Duplicate hash entry: {name}')
        hashed_paths.add(name)
        if verify_hashes:
            require(hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, f'Hash mismatch: {name}')
    for folder in ('configs/slt2026', 'datasets/slt2026', 'manifests/slt2026'):
        actual = {str(p.relative_to(root)) for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl')}
        require(actual <= hashed_paths, f'Unhashed release file in {folder}')
    systems = indexed(read_json(root/'configs/slt2026/systems.json')['systems'], 'id')
    require(len(systems)==12 and 'vits_el_1gpu' not in systems, 'Paper system roster mismatch')
    require(set(systems)==set(protocol['system_ids']), 'Protocol/system registry mismatch')
    refs = indexed(read_json(root/'configs/slt2026/reference_assets.json')['assets'], 'id')
    models = indexed(read_json(root/'configs/slt2026/model_assets.json')['assets'], 'id')
    require(len(models)==8 and len(refs)==83, 'Model/reference registry counts differ')
    for ref in refs.values():
        relative(ref['path'])
    source, variants = {}, {}
    word_count_warnings = 0
    for path in sorted((root/'datasets/slt2026/source').glob('*.json')):
        issues = validate_dataset_file(path)
        require(not [issue for issue in issues if issue.severity=='error'], f'Invalid dataset: {path}')
        word_count_warnings += sum(issue.code=='word_count_mismatch' for issue in issues)
        data = read_json(path)
        require(data['total_sentences']==20 and len(data['sentences'])==20, 'Scenario count mismatch')
        for s in data['sentences']:
            identity = data['dataset_id'], s['id']
            require(identity not in source, 'Duplicate source sentence')
            source[identity] = (data['scenario'], s, path.name)
        for method in ('raw','regex','llm'):
            normalized = read_json(root/'datasets/slt2026/normalized'/method/path.name)
            require(normalized['total_sentences']==20 and len(normalized['sentences'])==20, 'Normalized count mismatch')
            require(normalized['preprocessing']['method']==method, 'Normalization method mismatch')
            for s in normalized['sentences']:
                identity = data['dataset_id'], s['id']
                require(identity in source and s['original_text']==source[identity][1]['text'], 'Normalized/source text mismatch')
                require(identity+(method,) not in variants, 'Duplicate normalized sentence')
                variants[identity+(method,)] = s['normalized_text']
    require(len(source)==140 and len(variants)==420, 'Source/variant counts differ')
    require(Counter(v[0] for v in source.values()) == Counter({s:20 for s in {v[0] for v in source.values()}}), 'Scenario imbalance')
    require(len({v[0] for v in source.values()})==7, 'Scenario roster mismatch')
    require({(r['dataset_id'],r['sentence_id']) for r in protocol['sentences']} == set(source), 'Pinned sentence IDs differ')
    report = {'sentences':len(source), 'normalized_inputs':len(variants), 'legacy_word_count_warnings':word_count_warnings}
    vits_assignment = {}
    for track, expected_rows in [('core',5040),('heldout',560)]:
        rows = read_jsonl(root/f'manifests/slt2026/synthesis_{track}.jsonl')
        by_id = indexed(rows)
        require(len(rows)==expected_rows, f'{track} clip count mismatch')
        require(set(r['tts_system_id'] for r in rows)==set(systems), f'{track} system roster mismatch')
        require(Counter(r['tts_system_id'] for r in rows)==Counter({sid:system[f'{track}_clips'] for sid,system in systems.items()}), 'Per-system counts differ')
        combos = set()
        for row in rows:
            sid = row['tts_system_id']
            identity = row['dataset_id'],row['sentence_id']
            combination = identity+(row['preprocessing_method'],sid)
            require(combination not in combos, 'Duplicate sentence/condition/system')
            combos.add(combination)
            relative(row['audio_path'])
            relative(row['reference_audio_path'])
            require(row['reference_asset_id'] in refs, 'Unknown reference asset')
            require(row['reference_audio_path']==refs[row['reference_asset_id']]['path'], 'Reference asset path mismatch')
            require(systems[sid]['model_asset_id'] in models, 'Unknown model asset')
            require(row['metadata']['model_ref']==systems[sid]['model_ref'], 'Model reference mismatch')
            require(row['metadata']['sample_rate_hz']==systems[sid]['sample_rate_hz'], 'Sample rate mismatch')
            require(row['handoff_text']==' '.join(row['synthesis_text'].split()), 'Handoff whitespace mapping mismatch')
            if track=='core':
                require(identity in source, 'Unknown core sentence')
                require(row['source_text']==source[identity][1]['text'], 'Core source text mismatch')
                require(row['scenario']==source[identity][0] and row['dataset_file']==source[identity][2], 'Scenario/source file mismatch')
                require(row['synthesis_text']==variants[identity+(row['preprocessing_method'],)], 'Core normalized text mismatch')
            else:
                require(row['preprocessing_method']=='raw' and row['source_text']==row['synthesis_text'], 'Held-out raw text changed')
                if sid=='greek_male_lora_colab':
                    require(row['reference_asset_id']=='colab_male' and row['reference_kind']=='self_target_speaker', 'Colab reference mapping mismatch')
                else:
                    require(row['reference_asset_id']=='natural_'+row['sentence_id'], 'Held-out natural reference mismatch')
            if sid=='vits_el_4gpu':
                speaker=row['speaker']
                require(speaker in (0,1), 'Unknown VITS speaker')
                require(row['paper_label']==('vits_F' if speaker==0 else 'vits_M'), 'VITS paper label mismatch')
                if track=='core':
                    require(row['reference_asset_id']==('shared_female' if speaker==0 else 'shared_male'), 'VITS reference gender mismatch')
                    require(identity not in vits_assignment or vits_assignment[identity]==speaker, 'VITS speaker changed across conditions')
                    vits_assignment[identity]=speaker
                else:
                    require(row['study_gender']==('female' if speaker==0 else 'male'), 'Held-out VITS assignment mismatch')
        if track=='core':
            expected={(d,s,m,sysid) for d,s in source for m in ('raw','regex','llm') for sysid in systems}
            require(combos==expected, 'Incomplete core Cartesian product')
            require(Counter(vits_assignment.values())==Counter({0:70,1:70}), 'VITS sentence split mismatch')
            for scenario in {v[0] for v in source.values()}:
                require(Counter(speaker for identity,speaker in vits_assignment.items() if source[identity][0]==scenario)==Counter({0:10,1:10}), 'VITS scenario balance mismatch')
        else:
            sentences={r['sentence_id'] for r in rows}
            require(len(sentences)==80, 'Held-out sentence count mismatch')
            texts={}
            for row in rows:
                require(row['sentence_id'] not in texts or texts[row['sentence_id']]==row['source_text'], 'Held-out source texts differ by system')
                texts[row['sentence_id']]=row['source_text']
            require(Counter(r['study_gender'] for r in rows if r['tts_system_id']=='vits_el_4gpu')==Counter({'female':40,'male':40}), 'Held-out gender balance mismatch')
        transcription = indexed(read_jsonl(root/f'manifests/slt2026/transcription_{track}.jsonl'))
        references = indexed(read_jsonl(root/f'manifests/slt2026/speaker_references_{track}.jsonl'))
        require(set(transcription)==set(by_id)==set(references), f'{track} manifest join mismatch')
        for uid,row in by_id.items():
            trans=transcription[uid]
            require(trans['expected_text']==row['synthesis_text'] and trans['audio_path']==row['audio_path'], 'ASR join/text mismatch')
            require(trans['asr_id']=='whisperx_large_v3_el_noalign' and '--no_align' in trans['command'], 'ASR settings mismatch')
            relative(trans['transcript_path'])
            require(all(row[k]==value for k,value in references[uid].items()), 'Speaker-reference manifest join mismatch')
        report[track+'_clips']=len(rows)
    report['model_assets']=len(models)
    report['reference_assets']=len(refs)
    report['hashes_checked']=len(hashed_paths) if verify_hashes else 0
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    try:
        print(json.dumps(validate(args.root.resolve()),indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1,f'Input validation failed: {exc}\n')
