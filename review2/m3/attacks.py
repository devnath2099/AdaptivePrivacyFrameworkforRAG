"""Deterministic reversible-style obfuscations applied inside annotated spans only.

Labels denote the underlying entity, not validity under a formatting standard.
Attacks are controlled simulations, not a claim of semantic human validation.
"""
import copy
from review2.common import digest

ATTACKS = ('character_substitution', 'separator_insertion', 'whitespace', 'symbol_replacement',
           'email_obfuscation', 'phone_formatting', 'digit_words', 'case', 'token_splitting')


def transform(value, kind, attack):
    if attack == 'character_substitution':
        return value.translate(str.maketrans({'a': 'а', 'e': 'е', 'o': 'о'}))  # Cyrillic lookalikes
    if attack == 'separator_insertion':
        return '\u200b'.join(value)
    if attack == 'whitespace':
        return value.replace(' ', '  ')
    if attack == 'symbol_replacement':
        return value.translate(str.maketrans({'@': '＠', '.': '．', '-': '－'}))
    if attack == 'email_obfuscation':
        return value.replace('@', ' [at] ').replace('.', ' [dot] ') if 'EMAIL' in kind else value
    if attack == 'phone_formatting':
        if 'PHONE' not in kind:
            return value
        digits = ''.join(c for c in value if c.isdigit())
        return ('+' if value.startswith('+') else '') + ' '.join(digits[i:i + 3] for i in range(0, len(digits), 3))
    if attack == 'digit_words':
        words = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
        return ''.join(' ' + words[int(c)] + ' ' if c in '0123456789' else c for c in value).strip()
    if attack == 'case':
        # Case-sensitive credentials are excluded to avoid changing their identity.
        return value if any(k in kind for k in ('PASSWORD', 'KEY', 'TOKEN', 'PIN')) else value.swapcase()
    if attack == 'token_splitting':
        return ' '.join(value)
    raise ValueError('Unknown attack')


def attack_record(row, attack):
    result = copy.deepcopy(row)
    chunks, spans, cursor, position = [], [], 0, 0
    changes = []
    for span in row['canonical_spans']:
        prefix = row['text'][cursor:span['start']]
        value = row['text'][span['start']:span['end']]
        replacement = transform(value, span['type'], attack)
        chunks.extend([prefix, replacement])
        position += len(prefix)
        spans.append({'start': position, 'end': position + len(replacement), 'type': span['type'], 'value': replacement})
        position += len(replacement)
        cursor = span['end']
        changes.append(value != replacement)
    chunks.append(row['text'][cursor:])
    result.update(text=''.join(chunks), canonical_spans=spans, source_annotations=spans,
                  parent_record_id=row['record_id'], record_id=digest([row['record_id'], attack])[:24])
    result['metadata'].update(attack=attack, changed=any(changes), entity_changed=changes)
    return result


def attack_set(rows, attacks=ATTACKS):
    return [attacked for row in rows for name in attacks
            if (attacked := attack_record(row, name))['metadata']['changed']]
