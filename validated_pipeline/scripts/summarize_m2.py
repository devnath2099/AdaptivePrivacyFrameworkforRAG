"""Print concise M2 readiness diagnostics; reads only saved train artifacts."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from validated.common import path, read_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='Project-relative run directory')
    args = parser.parse_args()
    report = read_json(path(args.run) / 'labels' / 'train_diagnostics.json')
    for target, detail in report['diagnostics'].items():
        data = detail['all']
        print(f'\n{target}: abstain={data["all_abstain_rate"]:.2%}, conflict={data["conflict"]:.2%}')
        for name, support in data['class_support'].items():
            print(f'  {name}: LF-votes={support["lf_firing_count"]}, '
                  f'majority={support["majority_argmax_support"]}, '
                  f'final={support["final_argmax_support"]}, '
                  f'suppressed={support["aggregation_suppressed_by_snorkel"]}')
        print(f'  aggregation={data["aggregation_provenance"]}')
