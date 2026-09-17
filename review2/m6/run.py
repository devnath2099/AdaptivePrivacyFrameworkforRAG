from pathlib import Path
from review2.common import load_stage, read_json, read_jsonl, write_json, write_jsonl, digest, finish_stage
from .policies import generate_candidates, repository
from .profiling import profile
from .selection import select
from .experiments import compare, sensitivity


def run(config, run_dir):
    run_dir = Path(run_dir)
    parent = load_stage(run_dir, 'm5')
    cfg = config['m6']
    policies = repository(cfg['policies'] if 'policies' in cfg else generate_candidates(cfg['space']))
    observations = read_json(cfg['observations'])
    profiles = profile(observations, policies, cfg['allow_controlled'])
    if set(policies) != set(profiles):
        raise ValueError('All comparison policies need validation profiles')
    rows = read_jsonl(run_dir / 'm5/risk.jsonl')
    decisions = [select(r, policies, profiles, cfg) for r in rows]
    directory = run_dir / 'm6'
    write_json(directory / 'policies.json', policies)
    write_json(directory / 'profiles.json', profiles)
    write_jsonl(directory / 'decisions.jsonl', decisions)
    write_json(directory / 'comparison.json', compare(rows, policies, profiles, cfg))
    write_json(directory / 'sensitivity.json', sensitivity(rows, policies, profiles, cfg))
    # Controlled cases ensure the optimizer is exercised even with a weak detector.
    cases = [{'record_id': f'controlled_{i}', 'split': 'validation', 'score': score, 'uncertainty': u,
              'ignorance': 0., 'entity_types': ['EMAIL']} for i, (score, u) in enumerate([(0., 0.), (.2, .1), (.5, .1), (.9, .9)])]
    write_json(directory / 'controlled_comparison.json', compare(cases, policies, profiles, cfg))
    behavior = {}
    for method in ('max', 'weighted', 'ds'):
        risks = read_jsonl(run_dir / f'm5/{method}_risk.jsonl')
        chosen = [select(r, policies, profiles, cfg) for r in risks]
        behavior[method] = {'fallbacks': sum(d['fallback'] for d in chosen),
                            'policies': [d['policy']['id'] if d['policy'] else 'block' for d in chosen]}
    write_json(directory / 'risk_method_policy_comparison.json', behavior)
    return finish_stage(directory, config, digest(parent), enforcement_implemented=False,
                        profile_scope='controlled' if cfg['allow_controlled'] else 'measured_validation')


if __name__ == '__main__':
    from review2.pipeline import module_cli
    module_cli('m6')
