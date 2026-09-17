from pathlib import Path
from transformers import AutoModelForTokenClassification, AutoTokenizer
from review2.common import load_stage, read_json, read_jsonl, write_json, write_jsonl, digest, finish_stage, seed_all, evaluation_cohort
from review2.m2.alignment import encode
from review2.m2.training import train
from .attacks import attack_set
from .evaluation import compare


def run(config, run_dir):
    run_dir = Path(run_dir)
    m1 = load_stage(run_dir, 'm1')
    parent = load_stage(run_dir, 'm2', digest(m1))
    selected = read_json(run_dir / 'm2/selected.json')
    model_path = run_dir / 'm2' / selected['id'] / 'model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    labels = read_json(run_dir / 'm1/labels.json')
    length = selected['max_length']
    rows = {s: read_jsonl(run_dir / f'm1/{s}.jsonl') for s in ('train', 'validation')}
    rows['validation'] = evaluation_cohort(rows['validation'], config.get('evaluation', {}).get('m3_validation'),
                                         config['seed'], run_dir / 'm3/validation_cohort.json')
    attacked = attack_set(rows['validation'], config['m3']['attacks'])
    if not attacked:
        raise ValueError('No applicable validation attacks')
    data = {s: encode(r, tokenizer, labels, length) for s, r in rows.items()}
    adv_data = encode(attacked, tokenizer, labels, length)
    directory = run_dir / 'm3'
    write_jsonl(directory / 'validation_attacks.jsonl', attacked)
    comparisons = []
    clean_training = read_json(run_dir / 'm2' / selected['id'] / 'training.json')
    clean_model = AutoModelForTokenClassification.from_pretrained(model_path).to(config['device'])
    result, predictions = compare(clean_model, rows['validation'], tokenizer, labels, length,
                                  config['training']['batch_size'], config['m3']['attacks'])
    write_json(directory / 'clean_detector.json', result)
    write_jsonl(directory / 'clean_predictions.jsonl', predictions)
    del clean_model
    # Matched extra-training control prevents attributing another epoch's gains to FGSM.
    settings = [(0., 0.)] + [(e, l) for e in config['m3']['epsilons'] for l in config['m3']['lambdas']]
    for epsilon, coefficient in settings:
        seed_all(config['seed'])
        model = AutoModelForTokenClassification.from_pretrained(model_path).to(config['device'])
        name = 'standard_continuation' if coefficient == 0 else f'fgsm_e{epsilon:g}_l{coefficient:g}'
        cfg = {**config['training'], 'epochs': config['m3']['epochs'], 'loss': selected['id'].rsplit('_', 1)[1], 'seed': config['seed']}
        contract = {'parent': digest(parent), 'epsilon': epsilon, 'lambda_adv': coefficient, 'training': cfg, 'stage': 'm3'}
        info = train(model, tokenizer, labels, data['train'], data['validation'], cfg, directory / name, contract,
                     {'epsilon': epsilon, 'lambda_adv': coefficient, 'validation_data': adv_data})
        robust, predictions = compare(model, rows['validation'], tokenizer, labels, length, cfg['batch_size'], config['m3']['attacks'])
        write_json(directory / name / 'robustness.json', robust)
        write_jsonl(directory / name / 'predictions.jsonl', predictions)
        comparisons.append({'id': name, 'epsilon': epsilon, 'lambda_adv': coefficient, **info,
                            'additional_training_over_clean_ratio': info['training_seconds'] / max(clean_training['training_seconds'], 1e-9),
                            'clean_f1_change': robust['clean']['f1'] - result['clean']['f1'],
                            'inference_seconds_change': robust['clean']['seconds'] - result['clean']['seconds'],
                            'results': robust})
        del model
    control = comparisons[0]
    for item in comparisons[1:]:
        item['matched_standard_control'] = {
            'clean_f1_change': item['results']['clean']['f1'] - control['results']['clean']['f1'],
            'training_overhead_ratio': item['training_seconds'] / max(control['training_seconds'], 1e-9),
            'attack_f1_changes': {name: result['f1'] - control['results']['attacks'][name]['f1']
                                  for name, result in item['results']['attacks'].items() if 'f1' in result}}
    best = max((r for r in comparisons if r['epsilon'] > 0 and r['lambda_adv'] > 0), key=lambda r: r['best_validation_score'])
    write_json(directory / 'comparison.json', comparisons)
    write_json(directory / 'selected.json', {'id': best['id'], 'max_length': length,
               'selection': 'mean clean and pooled textual-attack validation strict F1'})
    return finish_stage(directory, config, digest(parent))


if __name__ == '__main__':
    from review2.pipeline import module_cli
    module_cli('m3')
