from .attacks import attack_record
from review2.m2.alignment import encode
from review2.m2.inference import evaluate
from review2.m2.metrics import entity_metrics


def compare(model, rows, tokenizer, labels, length, batch_size, attacks):
    clean, clean_pred, _ = evaluate(model, encode(rows, tokenizer, labels, length), tokenizer, labels, batch_size)
    results, saved = {}, []
    for name in attacks:
        pairs = [(i, attack_record(row, name)) for i, row in enumerate(rows)]
        pairs = [(i, row) for i, row in pairs if row['metadata']['changed']]
        if not pairs:
            results[name] = {'status': 'no_applicable_entities'}
            continue
        attacked = [row for _, row in pairs]
        metrics, pred, _ = evaluate(model, encode(attacked, tokenizer, labels, length), tokenizer, labels, batch_size)
        susceptible = successes = 0
        for (index, row), entities in zip(pairs, pred):
            original_gold = rows[index]['canonical_spans']
            before = {(s['start'], s['end'], s['type']) for s in clean_pred[index]}
            after = {(s['start'], s['end'], s['type']) for s in entities}
            for old, new, changed in zip(original_gold, row['canonical_spans'], row['metadata']['entity_changed']):
                if changed and (old['start'], old['end'], old['type']) in before:
                    susceptible += 1
                    successes += (new['start'], new['end'], new['type']) not in after
            saved.append({'record_id': row['record_id'], 'parent_record_id': row['parent_record_id'],
                          'split': row['split'], 'attack': name, 'entities': entities})
        results[name] = {**metrics, 'attack_success_rate': successes / susceptible if susceptible else None,
                         'eligible_previously_correct_entities': susceptible,
                         'f1_degradation_from_all_clean': clean['f1'] - metrics['f1']}
        paired_clean = entity_metrics([rows[i] for i, _ in pairs], [clean_pred[i] for i, _ in pairs])
        results[name]['paired_clean_f1'] = paired_clean['f1']
        results[name]['paired_f1_degradation'] = paired_clean['f1'] - metrics['f1']
    return {'clean': clean, 'attacks': results}, saved
