"""Local step before any V20 GPU work: write temp/v20/lock.json and the upload archive.

python -X utf8 -B temp/v20/freeze.py scripts     upload.zip with the V20 scripts and the plan (for the CPU dry run)
python -X utf8 -B temp/v20/freeze.py lock        lock.json: hashes of the scripts, the plan, the called modules,
                                                  the four cohort files and the two earlier protocols

Runs in the local research workspace (D:). It reads no reserved test split.
"""
from pathlib import Path
import ast
import hashlib
import json
import sys
import time
import zipfile

R = Path(__file__).resolve().parents[2]
P = R / 'deliverables/20261113/Numeric_Fidelity_Reviewed_v5_20261113'
PLAN = 'docs/superpowers/plans/2026-11-13-seed-extension-crossed-choice.md'
SCRIPTS = ('common.py', 'preflight.py', 'evaluate.py', 'seed_train.py', 'run_queue.py', 'collect.py', 'analysis.py', 'peek.py')
MODULES = ('v13_training', 'v14_training', 'v13_data', 'v13_protocol', 'v13_objectives', 'v14_objectives', 'v12_reader',
           'v12_protocol', 'v11_training', 'v11_pipeline', 'v11_model', 'v11_error_analysis', 'v4_model', 'v9_model',
           'model', 'benchmark', 'repro_probe', 'v16_code_fidelity')
DATA = {
    'development_crossed': (R / 'data/v16-role-balance-20261106/development.jsonl', 'data/v16-role-balance-20261106/development.jsonl'),
    'new_scene_crossed': (R / 'data/v19-prospective-20261108/crossed.jsonl', 'data/v19-prospective-20261108/crossed.jsonl'),
    'new_scene_original': (R / 'data/v19-prospective-20261108/original.jsonl', 'data/v19-prospective-20261108/original.jsonl'),
    'development_original': (P / 'data/v13-answerable-relations-20261026/development.jsonl',
                             'data/v13-answerable-relations-20261026/development.jsonl'),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(step):
    new = [R / 'temp/v20' / n for n in SCRIPTS] + [R / PLAN]
    for p in new:
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf8'), filename=str(p))
    if step == 'scripts':
        archive = R / 'temp/v20/upload.zip'
        assert not archive.exists()
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for p in new:
                z.write(p, p.relative_to(R).as_posix())
        receipt = dict(sha256=sha(archive), bytes=archive.stat().st_size, files=len(new), time=time.time(),
                       file_sha256={p.relative_to(R).as_posix(): sha(p) for p in new})
        (R / 'temp/v20/upload_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8')
        print(json.dumps(dict(sha256=receipt['sha256'], bytes=receipt['bytes'], files=receipt['files'])))
        return
    assert step == 'lock'
    modules = [R / 'src/binding' / (n + '.py') for n in MODULES]
    pins = json.loads((P / 'confirmation/protocol.json').read_text(encoding='utf8'))['pins']
    for p in modules:
        assert sha(p) == sha(P / 'code/src/binding' / p.name), p
        rel = p.relative_to(R).as_posix()
        assert pins.get(rel, sha(p)) == sha(p), rel
    uploaded = json.loads((R / 'temp/v20/upload_receipt.json').read_text(encoding='utf8'))['file_sha256']
    for p in new:
        assert uploaded[p.relative_to(R).as_posix()] == sha(p), f'{p} changed after the upload archive was built'
    lock = dict(
        date='2026-11-13', status='fixed_before_gpu', plan=PLAN,
        files={p.relative_to(R).as_posix(): sha(p) for p in new + modules},
        data_sha256={name: sha(local) for name, (local, _) in DATA.items()},
        data_paths={name: remote for name, (_, remote) in DATA.items()},
        original_protocol_sha256=sha(P / 'confirmation/protocol.json'),
        v19_prospective_protocol_sha256=sha(R / 'results/v19-budget-layout-20261108/prospective/protocol.json')
        if (R / 'results/v19-budget-layout-20261108/prospective/protocol.json').exists() else sys.argv[2],
        new_seeds=[101, 131, 151], updates=512, replay_updates=32, test_accessed=False, confirmation_used=False,
        created=time.time())
    path = R / 'temp/v20/lock.json'
    assert not path.exists()
    path.write_text(json.dumps(lock, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(lock_sha256=sha(path), files=len(lock['files']))))


if __name__ == '__main__':
    main(sys.argv[1])
