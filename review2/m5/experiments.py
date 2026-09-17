import copy
import time
from .aggregation import aggregate


def compare(records, config):
    reports, results = {}, {}
    for method in ('max', 'weighted', 'ds'):
        started = time.perf_counter()
        original = [aggregate(r, config, method) for r in records]
        elapsed = time.perf_counter() - started
        changes = []
        for shift in (-.1, .1):
            changed = copy.deepcopy(records)
            for record in changed:
                for entity in record['entities']:
                    entity['calibrated_confidence'] = max(0., min(1., entity['calibrated_confidence'] + shift))
            result = [aggregate(r, config, method) for r in changed]
            changes.append({'confidence_shift': shift, 'mean_absolute_score_change':
                            sum(abs(a['score'] - b['score']) for a, b in zip(original, result)) / max(1, len(result))})
        duplicated = []
        for r in records:
            duplicate = copy.deepcopy(r)
            duplicate['entities'] *= 2
            duplicated.append(aggregate(duplicate, config, method))
        reports[method] = {'seconds': elapsed, 'sensitivity': changes,
                           'duplicate_evidence_max_delta': max((abs(a['score'] - b['score']) for a, b in zip(original, duplicated)), default=0),
                           'mean_score': sum(r['score'] for r in original) / max(1, len(original)),
                           'conclusion': 'No independent risk gold or M7/M8 outcomes; no superiority claim'}
        results[method] = original
    return reports, results
