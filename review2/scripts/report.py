"""Summarize recorded results without inventing missing experiments."""
import xml.etree.ElementTree as ET
from pathlib import Path
from review2.common import ROOT, read_json, read_jsonl


def main():
    lines = ['# Actual execution report', '', 'Results below are read from saved artifacts, not expected outcomes.', '']
    xml = ROOT / 'outputs/tests.xml'
    if xml.exists():
        suites = ET.parse(xml).getroot().iter('testsuite')
        counts = {k: 0 for k in ('tests', 'failures', 'errors', 'skipped')}
        for suite in suites:
            for key in counts:
                counts[key] += int(suite.get(key, 0))
        lines += [f"Tests: {counts['tests']} executed; {counts['failures']} failures, {counts['errors']} errors, {counts['skipped']} skipped.",
                  'Artifacts: `outputs/tests.xml`, `outputs/pytest_result.txt`.', '']
    for name in ('verified_smoke', 'pretrained_cpu'):
        run = ROOT / 'outputs' / name
        if not run.exists():
            continue
        lines += [f'## {name}', '']
        if (run / 'status.json').exists():
            status = read_json(run / 'status.json')
            lines += [f"Completed stages: {', '.join(status['completed_stages'])}. Final test evaluated: {status['test_evaluated']}.", '']
        else:
            lines += ['Run has no completion marker. Inspect run.log and failure.json before claiming completion.', '']
        if (run / 'config.json').exists():
            cfg = read_json(run / 'config.json')
            lines += [f"Run kind: `{cfg['run_kind']}`. Dataset cap: {cfg['m1']['sources'][0].get('limit_per_split')} rows per official split.", '']
        if (run / 'm1/statistics.json').exists():
            stats = read_json(run / 'm1/statistics.json')
            audit = read_json(run / 'm1/annotation_integrity.json')
            lines += ['| Split | Records | Gold entities |', '|---|---:|---:|']
            for split, value in stats.items():
                lines.append(f"| {split} | {value['records']} | {sum(value['entities'].values())} |")
            lines += ['', f"Accepted {audit['accepted']}; quarantined {len(audit['rejected'])}; BIO repairs {audit.get('orphan_i_repairs', 0)}.",
                      f"Release/canonical inventory: {(len(read_json(run / 'm1/labels.json'))-1)//2} entity types.", '']
        if (run / 'm2/comparison.json').exists():
            lines += ['| M2 variant | Validation strict F1 | Recall | Inference ms/record | Training seconds |', '|---|---:|---:|---:|---:|']
            for r in read_json(run / 'm2/comparison.json'):
                lines.append(f"| {r['id']} | {r['f1']:.6f} | {r['recall']:.6f} | {r['milliseconds_per_record']:.2f} | {r['training_seconds']:.2f} |")
            lines += ['']
        if (run / 'm3/comparison.json').exists():
            lines += ['| M3 variant | Clean validation F1 | Mean attack F1 (applicable families) |', '|---|---:|---:|']
            for r in read_json(run / 'm3/comparison.json'):
                attacks = [a['f1'] for a in r['results']['attacks'].values() if 'f1' in a]
                lines.append(f"| {r['id']} | {r['results']['clean']['f1']:.6f} | {sum(attacks)/max(1,len(attacks)):.6f} |")
            lines += ['', 'Matched continuation control separates the extra training epoch from FGSM. Detailed paired attack metrics, eligible attack-success denominators and overhead are in `m3/`.', '']
        if (run / 'm4/comparison.json').exists():
            selection = read_json(run / 'm4/selected.json')
            temperature = read_json(run / 'm4/temperature.json')
            lines += [f"M4 selected `{selection['method']}` on validation. Fitted T={temperature['temperature']:.6f}; calibration NLL {temperature['nll_before']:.6f} → {temperature['nll_after']:.6f}.", '',
                      '| Reliability method | ECE | NLL | Brier | AURC | Seconds |', '|---|---:|---:|---:|---:|---:|']
            for name_, r in read_json(run / 'm4/comparison.json').items():
                m = r['metrics']
                lines.append(f"| {name_} | {m['ece']:.6f} | {m['nll']:.6f} | {m['brier']:.6f} | {m['aurc']:.6f} | {r['seconds']:.2f} |")
            lines += ['', 'These are token metrics including O; PII-only metrics are also saved. Plot: `m4/reliability.svg`.', '']
        if (run / 'm5/comparison.json').exists():
            lines += ['| Risk method | Mean validation risk score | Duplicate evidence max change |', '|---|---:|---:|']
            for method, r in read_json(run / 'm5/comparison.json').items():
                lines.append(f"| {method} | {r['mean_score']:.6f} | {r['duplicate_evidence_max_delta']:.6g} |")
            lines += ['', 'Risk scores use a disclosed category-ordering assumption. No risk accuracy/superiority claim is possible without independent outcomes.', '']
        if (run / 'm6/controlled_comparison.json').exists():
            lines += ['| Controlled policy mode | Constraint violations | Block fallbacks |', '|---|---:|---:|']
            for mode, r in read_json(run / 'm6/controlled_comparison.json').items():
                lines.append(f"| {mode} | {r['constraint_violations']} | {r['fallbacks']} |")
            rows = read_jsonl(run / 'm6/decisions.jsonl')
            lines += ['', f"Detector-derived validation decisions: {len(rows)}; block fallbacks: {sum(r['fallback'] for r in rows)}.",
                      'Controlled profile numbers are fixtures. Lower simulated cost/violations does not establish real privacy or RAG utility gains.', '']
        if (run / 'final_test/detector_comparison.json').exists():
            lines += ['| Frozen final-test M2 variant | Strict F1 | Recall |', '|---|---:|---:|']
            for variant, r in read_json(run / 'final_test/detector_comparison.json').items():
                lines.append(f"| {variant} | {r['f1']:.6f} | {r['recall']:.6f} |")
            lines += ['', 'Final test results were not used for tuning. Near-zero scores on these small runs are failures to establish useful detection, not successful thesis results.', '']
    lines += ['## Failures and limits', '',
              '- The first strict M1 pass quarantined WordPiece and orphan-I records; explicit audited normalization was added, and new runs were created.',
              '- The initial smoke attempt failed at M4 because Matplotlib was absent. The SVG writer removes that dependency; the failure remains in `outputs/smoke/failure.json`.',
              '- Python could not launch inside the sandbox; the existing environment worked with approved execution outside it. CPU-only PyTorch was used.',
              '- The complete million-record benchmark and large hyperparameter sweeps were not run. Full configuration is supplied, with memory/storage requirements documented in README.',
              '- Full inference currently truncates to the selected maximum length rather than implementing a new sliding-window methodology. Coverage loss is measured and reported.',
              '- M5 risk mapping and M6 controlled profiles need manual methodological review and future measured validation; they are explicit replaceable assumptions.',
              '- No M7–M9 code, mechanism enforcement, local RAG, or real end-to-end privacy/utility result is claimed.',
              '', '## Artifacts and tree', '',
              'All run artifacts are below `review2/outputs/`. `outputs/application_corpora/` contains 200 reused medical/public application records with no PII supervision claim.',
              'See `TREE.txt` for the source tree and artifact layout. Checkpoint files are retained; outputs are git-ignored to avoid committing datasets and multi-GB weights.', '']
    (ROOT / 'docs/RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    tree = ['review2/']
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if any(part in ('__pycache__', '.pytest_cache', 'outputs') for part in relative.parts):
            continue
        tree.append('  ' * len(relative.parts) + path.name + ('/' if path.is_dir() else ''))
    tree += ['', 'Generated artifacts (actual directories):']
    for run in sorted((ROOT / 'outputs').iterdir()):
        if run.is_dir() and not run.name.startswith('pytest'):
            tree.append(f'outputs/{run.name}/')
            for child in sorted(run.iterdir()):
                tree.append('  ' + child.name + ('/' if child.is_dir() else ''))
                if child.is_dir() and child.name in ('m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'final_test'):
                    for item in sorted(child.iterdir()):
                        tree.append('    ' + item.name + ('/' if item.is_dir() else ''))
    (ROOT / 'docs/TREE.txt').write_text('\n'.join(tree) + '\n', encoding='utf-8')
    print('Wrote docs/RESULTS.md and docs/TREE.txt')


if __name__ == '__main__':
    main()
