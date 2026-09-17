"""Create a code-only upload archive and a Kaggle notebook; no training artifacts."""
import ast
import json
import textwrap
import zipfile
from pathlib import Path
from review2.common import ROOT


def main():
    cells = []
    def markdown(text):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(text).strip()})
    def code(text):
        source = textwrap.dedent(text).strip()
        ast.parse(source)
        cells.append({'cell_type': 'code', 'metadata': {}, 'source': source, 'outputs': [], 'execution_count': None})

    markdown('''
        # Review-2: M1 → M6 on Kaggle

        1. Upload `review2_kaggle_bundle.zip` as a **private Kaggle Dataset**.
        2. Import this notebook and attach that dataset through **Add Input**.
        3. Enable a **GPU accelerator** and **Internet**, then run the cells in order.

        This runs actual pretrained DeBERTa and DistilBERT. The default acquisition is
        **50,000 valid records total**, not the entire benchmark. Evaluation uses fixed bounded cohorts. All six modules run.
        M5 mappings and M6 profiles remain explicitly controlled assumptions; no M7–M9
        enforcement or RAG runs. This notebook uses one GPU even if two are allocated.
        Save a notebook version with outputs to retain checkpoints and results.
    ''')
    code('''
        import os
        import sys
        import shutil
        import zipfile
        from pathlib import Path

        WORK = Path('/kaggle/working')
        INPUT = Path('/kaggle/input')
        assert WORK.exists() and INPUT.exists(), 'Run this notebook in Kaggle.'
        os.chdir(WORK)
        os.environ['HF_HOME'] = '/tmp/review2_huggingface'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        target = WORK / 'review2'
        sources = [p.parent for p in INPUT.rglob('pipeline.py') if p.parent.name == 'review2']
        archives = list(INPUT.rglob('review2_kaggle_bundle.zip'))
        if len(sources) == 1:
            shutil.copytree(sources[0], target, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('outputs', '__pycache__', '.pytest_cache'))
        elif len(archives) == 1 and not sources:
            with zipfile.ZipFile(archives[0]) as archive:
                for entry in archive.infolist():
                    (WORK / entry.filename).resolve().relative_to(WORK.resolve())
                archive.extractall(WORK)
        else:
            raise RuntimeError('Attach exactly one review2 code bundle using Add Input.')
        assert (target / 'pipeline.py').exists()
        sys.path.insert(0, str(WORK))
        print('Code:', target)
        print('Free working-disk GB:', round(shutil.disk_usage(WORK).free / 1e9, 1))
    ''')
    markdown('''
        Install the Python dependencies while preserving Kaggle's CUDA-enabled PyTorch.
        If Kaggle requests a session restart after installation, restart and rerun from
        the first cell. The upload archive contains no local virtual environment.
    ''')
    code('''
        import subprocess
        requirements = (target / 'requirements.txt').read_text().splitlines()
        dependencies = [r for r in requirements if r.strip() and not r.startswith('torch')]
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *dependencies], check=True)
        subprocess.run([sys.executable, '-c',
                        "import torch,transformers; print('torch',torch.__version__); print('transformers',transformers.__version__); print('GPU',torch.cuda.is_available()); assert torch.cuda.is_available(), 'Enable a GPU accelerator in notebook settings'"], check=True)
    ''')
    code('''
        import subprocess
        (target / 'outputs').mkdir(exist_ok=True)
        subprocess.run([sys.executable, '-m', 'pytest', 'review2/tests', '-q',
                        '--basetemp', 'review2/outputs/pytest_tmp',
                        '--junitxml=review2/outputs/tests.xml'], check=True)
    ''')
    markdown('''
        Choose settings **before** running. Use a new `RUN_NAME` whenever settings change.
        This 50k profile is not a four-hour runtime guarantee; do not set the cap to `None` on Kaggle without checking
        RAM, disk and session limits. The current implementation materializes data and MC
        arrays. Full checkpoints for all model/loss/control candidates also occupy many GB.
        `batch_size=4` is a conservative starting point, not a guarantee against GPU OOM.
    ''')
    code('''
        import json
        RUN_NAME = 'kaggle_review2_50k'
        PROFILE = 'kaggle_50k.json'
        # 50,000 valid records: 40,000 official train, 5,000 validation, 5,000 test.
        # About 20% of official train becomes calibration (group-preserving).
        # Fixed evaluation caps are documented in cfg['evaluation'].
        cfg = json.loads((target / 'configs' / PROFILE).read_text())
        cfg['device'] = 'cuda'
        cfg['threads'] = 2
        cfg['m6']['observations'] = str(target / 'configs/controlled_profiles.json')

        RUN = target / 'outputs' / RUN_NAME
        CONFIG = target / 'configs' / (RUN_NAME + '.json')
        if (RUN / 'config.json').exists():
            previous = json.loads((RUN / 'config.json').read_text())
            assert previous == cfg, 'Settings changed: choose a new RUN_NAME.'
        CONFIG.write_text(json.dumps(cfg, indent=2))
        print('Config:', CONFIG)
        print('Artifacts:', RUN)
    ''')
    markdown('''
        Run M1–M6 and then frozen test evaluation. This cell detects completed stage
        manifests and resumes from the first unfinished stage. It does not rerun completed
        upstream stages, which could change their artifact hashes. An interrupted training
        stage resumes its optimizer/RNG/batch checkpoint automatically.
    ''')
    code('''
        import subprocess
        from review2.common import load_stage, digest

        stages = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
        completed = 0
        parent = None
        for stage in stages:
            if not (RUN / stage / 'manifest.json').exists():
                break
            manifest = load_stage(RUN, stage, parent)
            parent = digest(manifest)
            completed += 1
        command = [sys.executable, '-u', '-m', 'review2.pipeline',
                   '--config', str(CONFIG), '--output', str(RUN)]
        if completed == len(stages):
            command += ['--test-only']
        else:
            command += ['--stages', *stages[completed:], '--evaluate-test']
        print(' '.join(command))
        with (WORK / (RUN_NAME + '_console.log')).open('a') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='')
                log.write(line)
                log.flush()
            returncode = process.wait()
        if returncode:
            raise RuntimeError(f'Pipeline exited with {returncode}; inspect the console and failure.json.')
    ''')
    code('''
        import json
        print(json.dumps(json.loads((RUN / 'status.json').read_text()), indent=2))
        for row in json.loads((RUN / 'm2/comparison.json').read_text()):
            print(row['id'], 'validation strict F1:', row['f1'],
                  'milliseconds/record:', round(row['milliseconds_per_record'], 2))
        print('Reliability selection:', json.loads((RUN / 'm4/selected.json').read_text()))
        print('Final test:', json.loads((RUN / 'final_test/detector_comparison.json').read_text()))
        from IPython.display import SVG, display
        display(SVG(filename=str(RUN / 'm4/reliability.svg')))
    ''')
    markdown('''
        ## Preserve and resume results

        Use **Save Version** with outputs retained, and download the run directory from
        notebook outputs. Keep `config.json`, all stage manifests, source cache, models,
        `best.pt` and `last.pt` if you need exact training resume. Do not prune files covered
        by a manifest: downstream stages validate their hashes.

        Across sessions, attach the prior saved output as an input and copy the complete
        run directory back to the same `/kaggle/working/review2/outputs/<RUN_NAME>` path
        before the configuration/run cells. Use the same source-code bundle and settings.
        Rerun the execution cell: it resumes an unfinished stage or uses `--test-only`
        for a frozen run. A GPU OOM requiring a different batch size needs a new run name.

        The M1 optional legacy medical/public corpus export is not required for M2–M6 and
        is not invoked here; those local caches were deliberately excluded from this bundle.
        M6 uses controlled policy-profile fixtures until actual M7/M8 validation data exists.
    ''')
    folder = ROOT / 'kaggle'
    folder.mkdir(exist_ok=True)
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                           'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 4}
    (folder / 'Review2_M1_M6_Kaggle.ipynb').write_text(json.dumps(notebook, indent=2), encoding='utf-8')
    archive_path = folder / 'review2_kaggle_bundle.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob('*')):
            relative = path.relative_to(ROOT)
            if not path.is_file() or any(p in {'outputs', 'kaggle', '__pycache__', '.pytest_cache'} for p in relative.parts):
                continue
            archive.write(path, Path('review2') / relative)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert 'review2/pipeline.py' in archive.namelist()
        assert not any('/outputs/' in name for name in archive.namelist())
    print('Notebook cells validated as Python; archive CRC verified.')
    print(folder / 'Review2_M1_M6_Kaggle.ipynb')
    print(archive_path)


if __name__ == '__main__':
    main()
