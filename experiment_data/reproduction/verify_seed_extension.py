"""Recompute the frozen seed-extension protocol and the additional seed and layout results."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import hashlib, json, re, sys, zipfile, xml.etree.ElementTree as ET
P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'reproduction'))
import v20_evidence_analysis as A
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))
F = P / 'followup_v20'
Z = F / 'evidence/completed_evidence.zip'
checks = []
# 1. archive, receipts, lock, plan, queue
host = read(F / 'evidence/evidence_receipt.json'); queue = read(F / 'evidence/queue_status.json')
assert sha(Z) == host['sha256'] == '89da11b97d4229748423be1a98c02d72d1bdbd5593fe73c5e7d696cf7c28efb6' and Z.stat().st_size == host['bytes'] == 5205861
lock_path = F / 'temp/v20/lock.json'; lock = read(lock_path)
assert sha(lock_path) == host['lock_sha256'] == '2f9a9c5e57fa41214a753759cb396144521cc404980fe16ecbad5acd0d0d159a' and lock['status'] == 'fixed_before_gpu' and lock['test_accessed'] is False
with zipfile.ZipFile(Z) as z:
    assert z.testzip() is None; names = z.namelist(); assert len(names) == host['entries'] == 175
    assert not any('heldout_test' in n or 'paired-v2-20261022/test' in n for n in names) and not any(n.endswith(('.safetensors', '.pt')) for n in names)
    assert hashlib.sha256(z.read('temp/v20/lock.json')).hexdigest() == sha(lock_path)
    for name, digest in lock['files'].items():
        if name in names:
            assert hashlib.sha256(z.read(name)).hexdigest() == digest, name
        if (F / name).exists():
            assert sha(F / name) == digest, name
    assert (F / lock['plan']).exists() and all((F / 'temp/v20' / n).exists() for n in ('common.py', 'preflight.py', 'evaluate.py', 'seed_train.py', 'run_queue.py', 'collect.py', 'analysis.py'))
    row_files = [n for n in names if n.endswith('rows.jsonl')]; pred_files = [n for n in names if n.endswith('predictions.jsonl')]
    rows = sum(len(z.read(n).splitlines()) for n in row_files)
jobs = [c['job'] for c in queue['completed']]
assert queue['phase'] == 'complete' and queue['failures'] == [] and all(c['exit_code'] == 0 for c in queue['completed'])
assert jobs == ['preflight', 'smoke', 'crossed-choice', 'replay-sft-seed73', 'replay-bottleneck-seed73'] + [f'main-compressed-{m}-seed{s}' for s in (101, 131, 151) for m in ('sft', 'bottleneck')] + ['new-seed-readouts', 'collect']
assert lock['created'] < queue['queue_started'] < queue['finished']
checks.append(f'Evidence archive matches the original host receipt; the lock, the plan and the V20 scripts match the lock; the queue finished all {len(jobs)} jobs without failure; {len(row_files)} row files with {rows} rows and {len(pred_files)} prediction files; no reserved test file and no model tensor')
# 2. recomputation
report, info = A.compute(P)
stored = read(F / 'analysis_v6/v20_numbers.json')
count = [0]
def same(x, y, path=''):
    if isinstance(x, dict):
        assert isinstance(y, dict) and set(x) == set(y), path
        for k in x: same(x[k], y[k], path + '/' + str(k))
    elif isinstance(x, list):
        assert isinstance(y, list) and len(x) == len(y), path
        for i, (a, b) in enumerate(zip(x, y)): same(a, b, f'{path}/{i}')
    elif isinstance(x, float) or isinstance(y, float):
        assert x is not None and y is not None and abs(x - y) < 1e-9, (path, x, y); count[0] += 1
    else:
        assert x == y, (path, x, y); count[0] += 1
same(json.loads(json.dumps(report)), stored['report'])
same(json.loads(json.dumps(info['posthoc'])), stored['record']['posthoc'])
PH = info['posthoc']
assert not info['differences_from_host_report'] and info['locked_files_checked'] >= 9
checks.append(f'{count[0]} stored values recomputed from the raw rows: the frozen analysis with the module pinned by the lock, equal to the report written on the host, and the post hoc figures with the function posthoc of v20_evidence_analysis.py')
# 3. protocol facts
pre = info['preflight']; R = report; B = R['part_b']; C = R['part_c']; A_ = R['part_a']
assert pre['status'] == 'pass' and pre['reserved_test_accessed'] is False and pre['frozen_before_new_inference'] and pre['lock_sha256'] == sha(lock_path) and pre['data_sha256'] == lock['data_sha256']
assert lock['created'] < pre['time'] < min(p['started_before_inference'] for p in info['protocols'].values())
assert all(p['model_selection'] is False and p['reserved_test_accessed'] is False for p in info['protocols'].values())
assert [len(info['protocols'][k]['jobs']) for k in ('smoke', 'crossed_choice', 'new_seeds')] == [1, 4, 3]
assert all(R['replay'][m]['exact'] and R['replay'][m]['pass'] and R['replay'][m]['updates'] == 32 for m in ('sft', 'bottleneck')) and len(info['adapters']) == 14
ck = B['checkpoints']
for s in (101, 131, 151):
    assert ck[f'sft-{s}']['initial_reader_identity'] == ck[f'bottleneck-{s}']['initial_reader_identity'] and ck[f'sft-{s}']['cache_sha256'] == ck[f'bottleneck-{s}']['cache_sha256'] == pre['seeds']['cache_sha256']
    assert sorted(ck[f'sft-{s}']['target_joint']) == ['128', '256', '512'] and ck[f'sft-{s}']['best_development_step'] == 512
assert len({ck[f'sft-{s}']['initial_reader_identity'] for s in (101, 131, 151)} | {pre['seeds']['original'][f'sft-{s}']['initial_reader_identity'] for s in (73, 97)}) == 5
checks.append('Preflight passed before any inference and after the lock; the replays of the two original seed-73 runs are exact over 32 updates; each new seed shares one initial reader and the original memory digest between the two arms; five distinct initializations')
print(json.dumps(dict(status="pass", stored_values_recomputed=count[0], checks=checks)))
