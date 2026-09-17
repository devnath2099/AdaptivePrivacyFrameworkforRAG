from pathlib import Path
from review2.common import load_stage, read_jsonl, write_json, write_jsonl, digest, finish_stage
from .experiments import compare
from .dempster_shafer import combine


def run(config, run_dir):
    run_dir = Path(run_dir)
    parent = load_stage(run_dir, 'm4')
    rows = read_jsonl(run_dir / 'm4/reliability.jsonl')
    if any(r['split'] != 'validation' for r in rows):
        raise ValueError('M5 development requires validation data')
    report, results = compare(rows, config['m5'])
    directory = run_dir / 'm5'
    write_json(directory / 'comparison.json', report)
    write_json(directory / 'conflict_experiment.json', [
        {'confidence': c, 'combination': combine({1: c, 15: 1-c}, {8: c, 15: 1-c}, config['m5']['conflict_limit'])}
        for c in (.5, .9, .99, 1.)])
    for method, records in results.items():
        write_jsonl(directory / f'{method}_risk.jsonl', records)
    write_jsonl(directory / 'risk.jsonl', results[config['m5']['method']])
    return finish_stage(directory, config, digest(parent), selection='Configured DS candidate; not declared best')


if __name__ == '__main__':
    from review2.pipeline import module_cli
    module_cli('m5')
