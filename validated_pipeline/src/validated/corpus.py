"""Strict natural adapters and deterministic exact-quota connected-group splitting."""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from collections import defaultdict

import numpy as np

from .common import digest, file_hash, path, read_json, write_json, metadata


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(text or ''))).strip()


def record(query, context, source, domain, rid, group, upstream_split):
    q, c = normalize(query), normalize(context)
    return dict(record_id=digest([source, upstream_split, str(rid)]), query_text=query,
                query_normalized_text=q, context_text=context, context_normalized_text=c,
                source_dataset=source, domain=domain, source_record_id=str(rid), group_id=str(group),
                source_split=upstream_split, label_source='natural_unlabeled', source_mode='natural')


def tatqa_rows(data, split):
    for doc in data:
        context = json.dumps({'table': doc['table']['table'], 'paragraphs': doc.get('paragraphs', [])}, ensure_ascii=False)
        group = doc['table'].get('uid') or digest(context)
        for q in doc['questions']:
            r = record(q['question'], context, 'tat_qa', 'financial_report', q['uid'], group, split)
            r['context_group'] = digest(context)
            yield r


def load_sources(manifest, caps):
    if manifest.get('allow_synthetic_fallback') is not False:
        raise ValueError('Source manifest must disable synthetic fallback')
    rows, audit = [], {}
    for name, spec in manifest['sources'].items():
        cache = path(f'cache/sources/{name}-{digest(spec)[:16]}.json')
        if cache.exists():
            saved = read_json(cache)
            if saved['spec'] != spec or saved['content_hash'] != digest(saved['records']):
                raise ValueError(f'Corrupt or incompatible source cache: {name}')
            loaded = saved['records']
        else:
            loaded = []
            if spec['adapter'] == 'tatqa_official':
                for split in spec['splits']:
                    url = f"https://raw.githubusercontent.com/{spec['dataset_id']}/{spec['revision']}/dataset_raw/tatqa_dataset_{split}.json"
                    with urllib.request.urlopen(url, timeout=120) as f:
                        loaded.extend(tatqa_rows(json.load(f), split))
            else:
                from datasets import load_dataset
                for split in spec['splits']:
                    ds = load_dataset(spec['dataset_id'], revision=spec['revision'], split=split,
                                      streaming=True, cache_dir=str(path('cache/huggingface')))
                    # Fixed candidate budget independent of requested scale; same pool across profiles.
                    for i, row in enumerate(ds):
                        q, c = row[spec['fields']['query']], row[spec['fields']['context']]
                        group = digest(normalize(c)) if name == 'natural_questions' and c else str(i)
                        loaded.append(record(q, c, name, spec['domain'], i, group, split))
                        if len(loaded) >= int(spec['cap']) * 2:
                            break
            if not loaded:
                raise ValueError(f'{name}: no natural source records loaded; no fallback permitted')
            write_json(cache, dict(spec=spec, content_hash=digest(loaded), records=loaded,
                                  provenance=metadata({'seed': manifest['seed'], 'source_manifest': manifest},
                                                      spec, len(loaded), spec['splits'])))
        rows.extend(loaded)
        audit[name] = dict(records_loaded=len(loaded), candidate_budget=spec['cap'] * 2,
                           cache_sha256=file_hash(cache), revision=spec['revision'], splits=spec['splits'])
    return rows, audit


def deduplicate_groups(records):
    """Union duplicate-query aliases before retaining one source-aware canonical record."""
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, r in enumerate(records):
        keys = [('query', normalize(r['query_normalized_text']).casefold()),
                ('group', r['source_dataset'], r['group_id'])]
        if r.get('context_group'):
            keys.append(('context', r['source_dataset'], r['context_group']))
        for key in keys:
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups, unique = defaultdict(list), {}
    for i in sorted(range(len(records)), key=lambda j: (records[j]['source_dataset'], records[j]['source_record_id'], records[j]['record_id'])):
        r = records[i]
        key = normalize(r['query_normalized_text']).casefold()
        if not key:
            continue
        if key not in unique:
            unique[key] = r
            groups[root(i)].append(dict(r))
    result = []
    for group in groups.values():
        gid = digest(sorted(r['record_id'] for r in group))
        for r in group:
            r['dedup_group_id'] = gid
        result.append(group)
    return result


def split_exact(records, caps, seed=42):
    """MILP over group-size signatures, not records; whole groups may be excluded to meet caps."""
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import lil_matrix
    names = sorted(caps)
    splits = ['train', 'calibration', 'validation', 'test']
    ratios = [0.7, 0.1, 0.1, 0.1]
    groups = deduplicate_groups(records)
    available = {s: sum(r['source_dataset'] == s for g in groups for r in g) for s in names}
    deficit = {s: int(caps[s]) - available[s] for s in names if available[s] < int(caps[s])}
    if deficit:
        raise ValueError(f'Natural query deficit after deduplication: {deficit}; available={available}; no substitution')
    buckets = defaultdict(list)
    for group in groups:
        signature = tuple(sum(r['source_dataset'] == s for r in group) for s in names)
        buckets[signature].append(group)
    signatures = sorted(buckets)
    n = len(signatures)
    a = lil_matrix((n + len(names) * 4, n * 4), dtype=float)
    lo, hi = np.zeros(a.shape[0]), np.zeros(a.shape[0])
    for j, sig in enumerate(signatures):
        hi[j] = len(buckets[sig])
        for k in range(4):
            a[j, j * 4 + k] = 1
            for s, count in enumerate(sig):
                a[n + s * 4 + k, j * 4 + k] = count
    for s, name in enumerate(names):
        for k, ratio in enumerate(ratios):
            target = int(caps[name]) * ratio
            if abs(target - round(target)) > 1e-6:
                raise ValueError('Caps must support exact 70/10/10/10 integer splits')
            lo[n + s * 4 + k] = hi[n + s * 4 + k] = round(target)
    solution = milp(np.random.default_rng(seed).uniform(0, 1, n * 4), integrality=np.ones(n * 4),
                    bounds=Bounds(0, np.inf), constraints=LinearConstraint(a.tocsr(), lo, hi),
                    options={'time_limit': 120})
    if not solution.success:
        raise ValueError(f'Cannot satisfy exact source quotas with intact groups: {solution.message}')
    out = {s: [] for s in splits}
    for j, sig in enumerate(signatures):
        pool = sorted(buckets[sig], key=lambda g: digest([seed, g[0]['dedup_group_id']]))
        offset = 0
        for k, split in enumerate(splits):
            count = int(round(solution.x[j * 4 + k]))
            for group in pool[offset:offset + count]:
                out[split].extend(dict(r, split=split) for r in group)
            offset += count
    for split in splits:
        out[split].sort(key=lambda r: digest([seed, r['record_id']]))
        for name in names:
            assert sum(r['source_dataset'] == name for r in out[split]) == round(caps[name] * ratios[splits.index(split)])
    check_isolation(out)
    return out


def check_isolation(splits):
    seen = {}
    for split, records in splits.items():
        for r in records:
            if r['source_mode'] != 'natural' and split != 'train':
                raise ValueError('Synthetic augmentation cannot enter non-training splits')
            for key in [('id', r['record_id']), ('q', normalize(r['query_normalized_text']).casefold()),
                        ('g', r.get('dedup_group_id', r['record_id']))]:
                if key in seen and seen[key] != split:
                    raise ValueError(f'Cross-split leakage: {key[0]}')
                seen[key] = split
