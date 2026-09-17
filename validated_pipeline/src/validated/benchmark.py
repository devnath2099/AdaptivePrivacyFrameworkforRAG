import yaml
from .common import path, digest
from .corpus import record, check_isolation

BASE = 'controlled_privacy_benchmark/'


def load(name):
    return yaml.safe_load(path(BASE + name).read_text(encoding='utf-8'))


def verify_separation():
    train, test = load('train_templates.yaml')['templates'], load('heldout_test_templates.yaml')['templates']
    for key in ['id', 'family', 'text']:
        if {t[key] for t in train} & {t[key] for t in test}:
            raise ValueError(f'Benchmark {key} overlap')
    lex = load('lexical_sets.yaml')
    for key in ['topic', 'medical', 'attribute', 'collection']:
        if set(lex['train'][key]) & set(lex['heldout'][key]):
            raise ValueError(f'Benchmark lexical overlap: {key}')
    if lex['train']['marker_prefix'] == lex['heldout']['marker_prefix']:
        raise ValueError('Placeholder lexicons overlap')


def generate(partition, count, tasks=None, scarce_only=False):
    if partition not in ['train', 'heldout'] or count < 0:
        raise ValueError('Invalid benchmark request')
    verify_separation()
    filename = 'train_templates.yaml' if partition == 'train' else 'heldout_test_templates.yaml'
    templates = load(filename)['templates']
    if scarce_only:
        templates = [t for t in templates if t['scenario'] in ['medical', 'financial', 'reid', 'attribute', 'membership']]
    lexical = load('lexical_sets.yaml')[partition]
    scenarios = load('scenario_definitions.yaml')['scenarios']
    if tasks is None:
        tasks = yaml.safe_load(path('configs/pipeline.yaml').read_text())['active_tasks']
    result = []
    for i in range(count):
        template = templates[i % len(templates)]
        values = {key: lexical[key][(i // len(templates)) % len(lexical[key])] for key in ['topic', 'medical', 'attribute', 'collection']}
        values['marker'] = f"[{lexical['marker_prefix']}_{i:05d}]"
        text = template['text'].format(**values)
        # Generic scenarios cycle a finite lexicon; emit each query once.
        if any(r['query_text'] == text for r in result):
            continue
        labels = scenarios[template['scenario']]
        r = record(text, '', 'controlled_ontology_v1', 'controlled', i, i, partition)
        r.update(scenario_id=template['scenario'], rule_id=labels['rule_id'], template_id=template['id'],
                 template_family=template['family'], lexical_set_id=lexical['id'], split=partition,
                 source_mode='synthetic_controlled', label_source='synthetic_verified',
                 label_provenance='operational_ontology_v1_by_construction',
                 future_risk_tier=labels['future_risk_tier'], future_expected_policy=labels['future_expected_policy'])
        r['targets'], r['observed'] = {}, {}
        for task, spec in tasks.items():
            if spec['kind'] == 'categorical':
                r['targets'][task] = [int(c == labels[task]) for c in spec['classes']]
                r['observed'][task] = 1.0
            else:
                r['targets'][task] = labels[task]
                r['observed'][task] = [1.0] * len(spec['labels'])
        result.append(r)
    return result


def augment(splits, config, tasks):
    if not config['enabled']:
        return splits
    n = int(config['count'])
    if n > int(config['max_synthetic_training_examples']) or n > len(splits['train']) * float(config['max_ratio_to_natural_train']):
        raise ValueError('Synthetic augmentation count/ratio exceeds configured cap')
    generated = generate('train', n, tasks, scarce_only=True)
    all_queries = {r['query_normalized_text'].casefold() for rows in splits.values() for r in rows}
    if any(r['query_normalized_text'].casefold() in all_queries for r in generated):
        raise ValueError('Synthetic query collides with natural corpus')
    result = {k: list(v) for k, v in splits.items()}
    result['train'] += generated
    check_isolation(result)
    return result
