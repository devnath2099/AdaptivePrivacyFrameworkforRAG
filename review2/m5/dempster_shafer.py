"""Four-element frame represented by bit masks; 15 denotes the entire frame."""
import math

FRAME = 15
TIERS = ('Low', 'Medium', 'High', 'Critical')


def validate_mass(mass):
    if not mass or any(not isinstance(k, int) or k < 1 or k > FRAME for k in mass):
        raise ValueError('Mass focal sets must be nonempty subsets of the frame')
    if any(not math.isfinite(v) or v < 0 for v in mass.values()) or abs(sum(mass.values()) - 1) > 1e-8:
        raise ValueError('Masses must be finite, nonnegative and normalized')


def combine(left, right, conflict_limit=1 - 1e-8):
    if not 0 < conflict_limit <= 1:
        raise ValueError('Conflict limit must be in (0,1]')
    validate_mass(left)
    validate_mass(right)
    result, conflict = {}, 0.
    for a, ma in left.items():
        for b, mb in right.items():
            intersection = a & b
            if intersection:
                result[intersection] = result.get(intersection, 0.) + ma * mb
            else:
                conflict += ma * mb
    if conflict >= conflict_limit:
        return {FRAME: 1.}, conflict, True
    result = {k: v / (1 - conflict) for k, v in result.items()}
    validate_mass(result)
    return result, conflict, False


def summarize(mass):
    validate_mass(mass)
    belief, plausibility, probability = [], [], []
    for i in range(4):
        bit = 1 << i
        belief.append(sum(v for k, v in mass.items() if k & bit == k))
        plausibility.append(sum(v for k, v in mass.items() if k & bit))
        probability.append(sum(v / k.bit_count() for k, v in mass.items() if k & bit))
    return {'belief': dict(zip(TIERS, belief)), 'plausibility': dict(zip(TIERS, plausibility)),
            'probabilities': probability, 'ignorance': mass.get(FRAME, 0.),
            'score': sum(i / 3 * p for i, p in enumerate(probability)),
            'tier': TIERS[max(range(4), key=lambda i: probability[i])]}
