"""Preserve source text: changing whitespace would invalidate character offsets."""
from review2.common import digest


def token_offsets(text, tokens, wordpiece=False):
    offsets, cursor = [], 0
    for token in tokens:
        continuation = wordpiece and token.startswith('##')
        if continuation:
            token = token[2:]
        start = text.find(token, cursor)
        if not token or start < 0 or text[cursor:start].strip():
            raise ValueError('Tokens cannot be aligned to source text without guessing')
        if continuation and start != cursor:
            raise ValueError('WordPiece continuation must be contiguous')
        offsets.append((start, start + len(token)))
        cursor = start + len(token)
    if text[cursor:].strip():
        raise ValueError('Source text contains unaligned trailing content')
    return offsets


def bio_spans(labels, offsets, repair_orphan=False):
    if len(labels) != len(offsets):
        raise ValueError('Token/label length mismatch')
    spans, current = [], None
    for label, (start, end) in zip(labels, offsets):
        if label == 'O':
            current = None
            continue
        if '-' not in label or label[0] not in 'BI':
            raise ValueError(f'Invalid BIO label: {label}')
        prefix, kind = label.split('-', 1)
        if prefix == 'I' and (current is None or current['type'] != kind):
            if not repair_orphan:
                raise ValueError('Orphan I label in source annotation')
            prefix = 'B'
        if prefix == 'B':
            current = {'start': start, 'end': end, 'type': kind}
            spans.append(current)
        else:
            current['end'] = end
    return spans


def canonical(row, source, source_id, split, type_map=None, wordpiece=False, repair_orphan=False):
    type_map = type_map or {}
    if 'tokens' in row:
        text = row.get('text') or ' '.join(row['tokens'])
        annotations = row['labels']
        spans = bio_spans(annotations, token_offsets(text, row['tokens'], wordpiece), repair_orphan)
        representation = 'native_text' if row.get('text') else 'whitespace_joined_tokens'
    else:
        text = row.get('text', row.get('source_text'))
        annotations = row.get('spans', row.get('privacy_mask', []))
        spans = [{'start': int(s['start']), 'end': int(s['end']), 'type': s.get('type', s.get('label')),
                  **({'value': s['value']} if 'value' in s else {})} for s in annotations]
        representation = 'native_text'
    for span in spans:
        span['type'] = type_map.get(span['type'], span['type'])
    record_id = digest([source, str(source_id)])[:24]
    return {'record_id': record_id, 'source_id': source, 'source_record_id': str(source_id),
            'text': text, 'source_annotations': annotations, 'canonical_spans': sorted(spans, key=lambda s: s['start']),
            'split': split, 'group_id': str(row.get('group_id', source_id)),
            'metadata': {'text_representation': representation,
                         'wordpiece_alignment': wordpiece and any(t.startswith('##') for t in row.get('tokens', [])),
                         'orphan_i_repair_enabled': repair_orphan,
                         'orphan_i_repairs': sum(label.startswith('I-') and (i == 0 or annotations[i-1] == 'O' or annotations[i-1][2:] != label[2:])
                                                 for i, label in enumerate(annotations)) if 'tokens' in row else 0}}
