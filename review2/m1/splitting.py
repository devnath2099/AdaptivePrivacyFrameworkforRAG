"""Union duplicates/groups, quarantine official-split conflicts, split whole components."""
from collections import defaultdict
from review2.common import digest


def deduplicate(rows):
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen, exact, discarded = {}, {}, []
    for i, row in enumerate(rows):
        for key in [('text', ' '.join(row['text'].casefold().split())),
                    ('group', row['source_id'], row['group_id'])]:
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[root(i)].append(row)
    retained = []
    for group in components.values():
        official = {r['split'] for r in group if r['split']}
        if len(official) > 1:
            discarded.extend({'record_id': r['record_id'], 'reason': 'official_split_conflict'} for r in group)
            continue
        gid = digest(sorted(r['record_id'] for r in group))
        for row in group:
            key = row['text']
            if key in exact:
                a = [(s['start'], s['end'], s['type']) for s in exact[key]['canonical_spans']]
                b = [(s['start'], s['end'], s['type']) for s in row['canonical_spans']]
                if a != b:
                    raise ValueError('Identical text has conflicting gold annotations')
                discarded.append({'record_id': row['record_id'], 'reason': 'exact_duplicate',
                                  'retained_id': exact[key]['record_id']})
                continue
            row = dict(row, component_id=gid, split=next(iter(official), None))
            exact[key] = row
            retained.append(row)
    return retained, discarded


def split_records(rows, seed, ratios, calibration_fraction=.15):
    if abs(sum(ratios.values()) - 1) > 1e-9 or any(v <= 0 for v in ratios.values()):
        raise ValueError('Split ratios must be positive and sum to one')
    groups = defaultdict(list)
    for row in rows:
        groups[row['component_id']].append(row)
    output = []
    for gid, group in sorted(groups.items()):
        # Source signature makes assignment source-aware while preserving mixed-source groups.
        value = int(digest([seed, sorted({r['source_id'] for r in group}), gid])[:16], 16) / 16**16
        split = group[0]['split']
        if split == 'train':
            split = 'calibration' if value < calibration_fraction else 'train'
        elif split is None:
            total = 0
            for name, ratio in ratios.items():
                total += ratio
                if value < total:
                    split = name
                    break
        output.extend(dict(r, split=split) for r in group)
    return output
