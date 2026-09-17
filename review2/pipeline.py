"""Run with python -m review2.pipeline --config ... --output ... ."""
import argparse
import importlib
import json
import logging
import platform
import sys
from pathlib import Path
import torch
from review2.common import read_json, write_json, seed_all, provenance, digest, ROOT

STAGES = ('m1', 'm2', 'm3', 'm4', 'm5', 'm6')


def execute(config, output, stages=STAGES, evaluate_test=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'final_test/frozen_selection.json').exists() and stages:
        raise ValueError('This run is frozen for final test; use a new output directory for further development')
    snapshot = output / 'config.json'
    if snapshot.exists() and read_json(snapshot) != config:
        raise ValueError('Output directory already belongs to a different configuration')
    write_json(snapshot, config)
    logging.basicConfig(filename=output / 'run.log', level=logging.INFO, force=True,
                        format='%(asctime)s %(levelname)s %(message)s', encoding='utf-8')
    torch.set_num_threads(config.get('threads', 2))
    seed_all(config['seed'])
    if config['device'] == 'auto':
        config = {**config, 'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    import transformers, datasets, numpy
    write_json(output / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
               'torch': torch.__version__, 'transformers': transformers.__version__, 'datasets': datasets.__version__,
               'numpy': numpy.__version__, 'cuda_available': torch.cuda.is_available(), **provenance(config)})
    for stage in stages:
        print(f'Running {stage}', flush=True)
        logging.info('Starting %s', stage)
        try:
            importlib.import_module(f'review2.{stage}.run').run(config, output)
        except Exception as error:
            logging.exception('Failed %s', stage)
            write_json(output / 'failure.json', {'stage': stage, 'type': type(error).__name__, 'message': str(error)})
            raise
        logging.info('Completed %s', stage)
    if evaluate_test:
        from review2.final_evaluation import evaluate
        try:
            evaluate(config, output)
        except Exception as error:
            logging.exception('Failed frozen final-test evaluation')
            write_json(output / 'failure.json', {'stage': 'final_test', 'type': type(error).__name__, 'message': str(error)})
            raise
    completed = [name for name in STAGES if (output / name / 'manifest.json').exists()]
    write_json(output / 'status.json', {'completed_stages': completed, 'test_evaluated': evaluate_test,
               'run_kind': config['run_kind'], 'enforcement_and_rag': 'out_of_scope'})


def module_cli(stage=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='review2/configs/smoke.json')
    parser.add_argument('--output', default='review2/outputs/smoke')
    parser.add_argument('--stages', nargs='+', choices=STAGES)
    parser.add_argument('--evaluate-test', action='store_true')
    parser.add_argument('--test-only', action='store_true', help='Resume/evaluate frozen test without rerunning development stages')
    args = parser.parse_args()
    if args.test_only and (stage or args.stages):
        parser.error('--test-only cannot be combined with module/stage selection')
    stages = () if args.test_only else (stage,) if stage else args.stages or STAGES
    execute(read_json(args.config), args.output, stages, args.evaluate_test or args.test_only)


if __name__ == '__main__':
    module_cli()
