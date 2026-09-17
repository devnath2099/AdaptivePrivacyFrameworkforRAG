"""Character-span to subword BIO alignment; special/padding positions never carry loss."""
import torch


def align(offsets, spans, label2id):
    labels, previous = [], None
    for start, end in offsets:
        if start == end:
            labels.append(-100)
            previous = None
            continue
        overlaps = [i for i, s in enumerate(spans) if start < s['end'] and end > s['start']]
        if not overlaps:
            labels.append(label2id['O'])
            previous = None
            continue
        if len(overlaps) != 1:
            raise ValueError('One token overlaps multiple gold entities')
        i = overlaps[0]
        span = spans[i]
        if start < span['start'] or end > span['end']:
            # A token straddling a boundary cannot express exact gold offsets; mask it.
            labels.append(-100)
            previous = None
            continue
        prefix = 'I' if previous == i else 'B'
        labels.append(label2id[f"{prefix}-{span['type']}"])
        previous = i
    return labels


def encode(rows, tokenizer, labels, max_length):
    mapping = {label: i for i, label in enumerate(labels)}
    output = []
    for row in rows:
        tokens = tokenizer(row['text'], truncation=True, max_length=max_length,
                           return_offsets_mapping=True, return_special_tokens_mask=True)
        offsets = tokens.pop('offset_mapping')
        tokens.pop('special_tokens_mask')
        tokens.pop('token_type_ids', None)
        output.append({**tokens, 'labels': align(offsets, row['canonical_spans'], mapping),
                       'offsets': offsets, 'record': row})
    return output


def collate(examples, pad_id):
    size = max(len(e['input_ids']) for e in examples)
    batch = {}
    for key, pad in [('input_ids', pad_id), ('attention_mask', 0), ('labels', -100)]:
        batch[key] = torch.tensor([e[key] + [pad] * (size - len(e[key])) for e in examples], dtype=torch.long)
    return batch


def decode(ids, offsets, labels, probabilities=None):
    spans, current = [], None
    for i, (index, (start, end)) in enumerate(zip(ids, offsets)):
        if start == end:
            current = None
            continue
        label = labels[int(index)]
        if label == 'O':
            current = None
            continue
        prefix, kind = label.split('-', 1)
        confidence = float(probabilities[i][int(index)]) if probabilities is not None else 1.0
        if prefix == 'B' or current is None or current['type'] != kind:
            current = {'start': start, 'end': end, 'type': kind, 'confidence': confidence, 'token_indices': [i]}
            spans.append(current)
        else:
            current['end'] = end
            current['confidence'] = min(current['confidence'], confidence)
            current['token_indices'].append(i)
    return spans
