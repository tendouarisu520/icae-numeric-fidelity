"""V20 preflight: CPU only, no CUDA. Pins data, readers and stored generations before any V20 GPU work.

python preflight.py          verifies the lock and writes results/v20-.../preflight.json (first job of the queue)
python preflight.py --dry    same checks without lock and without writing anything (used before the freeze)
"""
from common import *
import sys


def main(dry):
    environment()
    lock = None if dry else verify_lock()
    if not dry:
        assert not (O / 'preflight.json').exists(), 'Audit the earlier run; nothing is overwritten'
    for p in (RUNS, O / 'smoke', O / 'crossed_choice', O / 'new_seeds', O / 'analysis', O / 'completed_evidence.zip'):
        assert not p.exists(), f'{p} exists; V20 outputs are written once'
    original_path = R / 'results/confirmation-20261104/protocol.json'
    original = read(original_path)
    for name, h in original['pins'].items():
        assert sha(R / name) == h, name
    v19_path = R / 'results/v19-budget-layout-20261108/prospective/protocol.json'
    v19 = read(v19_path)
    jobs = [j for j in v19['jobs'] if j['label'] in SFT_LABELS]
    assert [j['label'] for j in jobs] == list(SFT_LABELS)
    for j in jobs:
        assert sha(R / j['checkpoint']) == j['checkpoint_sha256'] == original['pins'][j['checkpoint']], j['label']
        assert j['method'] == 'sft' and j['step'] == 512
    data = {name: sha(R / path) for name, path in COHORTS.items()}
    v19_pre = read(R / 'results/v19-budget-layout-20261108/preflight.json')
    assert data['new_scene_crossed'] == v19_pre['datasets']['crossed']['sha256'] == v19['data']['crossed']
    assert data['new_scene_original'] == v19_pre['datasets']['original']['sha256'] == v19['data']['original']
    assert data['development_crossed'] == v19_pre['old_cohort_hashes'][COHORTS['development_crossed']]
    assert data['development_original'] == v19_pre['old_cohort_hashes'][COHORTS['development_original']]
    manifest = read(R / 'data/v13-answerable-relations-20261026/manifest.json')
    assert data['development_original'] == manifest['partitions']['development']['sha256']
    train_path = R / 'data/v13-answerable-relations-20261026/train.jsonl'
    assert sha(train_path) == manifest['partitions']['train']['sha256']

    # Stored greedy outputs of the earlier crossed-view evaluations.
    references = {}
    for j in jobs:
        label = j['label']
        folders = {'development_crossed': R / 'results/v16-code-fidelity-20261106/roles' / label,
                   'new_scene_crossed': R / 'results/v19-budget-layout-20261108/prospective' / label / 'crossed'}
        references[label] = {}
        for name, folder in folders.items():
            receipt = read(folder / 'receipt.json')
            path = folder / 'predictions.jsonl'
            assert sha(path) == receipt['predictions_sha256'], path
            assert receipt['metrics']['rows'] == 2304 and receipt['metrics']['units'] == 144
            references[label][name] = dict(path=path.relative_to(R).as_posix(), sha256=receipt['predictions_sha256'],
                                           correct=receipt['metrics']['counts']['known_exact'][0])

    # CPU dry run of the data handling used by evaluate.py.
    from transformers import AutoTokenizer
    from binding.v13_data import prepare
    tokenizer = AutoTokenizer.from_pretrained(str(R / 'models/mistral'), local_files_only=True, use_fast=False)
    view = ('semantic_rotation', 'layout_rotation', 'target_team_position')
    shape = {}
    for name, path in COHORTS.items():
        units = rows(R / path)
        if name == 'development_original':
            units = [u for u in units if u['family'] == 'probe']
        identity = 0
        for u in units:
            item = prepare(u, tokenizer)
            assert u['family'] == 'probe' and len(u['contexts']) == 4 and len(u['queries']) == 4
            for i in range(4):
                a, b = item['answers'][i][:2]
                assert len(a) == len(b) and a != b and u['answers'][i][0] != u['answers'][i][1]
                assert len({f['value'] for f in u['facts'][i]}) == len(u['facts'][i])
            assert u['answers'][0][:2] == u['answers'][1][:2][::-1]
            identity += all(u.get(k, 0) == 0 for k in view)
        shape[name] = dict(units=len(units), scenes=len({u['source_id'] for u in units}),
                           articles=len({u['article'] for u in units}), identity_units=identity)
        if name.endswith('_crossed'):
            assert shape[name] == dict(units=144, scenes=8, articles=8, identity_units=8), shape[name]
            for label in SFT_LABELS:
                stored = {(r['unit_id'], r['context_index'], r['question_index']): r
                          for r in rows(R / references[label][name]['path'])}
                assert len(stored) == 2304
                for u in units:
                    for i in range(4):
                        for j in range(4):
                            assert stored[u['id'], i, j]['gold'] == u['answers'][i][j]
        else:
            assert shape[name]['units'] == 96 and shape[name]['identity_units'] == 96, shape[name]
    smoke = [u for u in rows(train_path) if u['family'] == 'probe'][:2]
    for u in smoke:
        prepare(u, tokenizer)

    # Seed extension: the extended schedule reproduces the original schedules; code and data are unchanged.
    import binding.v11_training as base
    by_source = {}
    for u in rows(train_path):
        by_source.setdefault(u['source_id'], []).append(u)
    ids = list(by_source)
    assert len(ids) == 512 and all(len(v) == 2 for v in by_source.values())
    new_schedule = schedule_v20()
    seeds = dict(original={}, new_schedule_sha256={})
    caches = set()
    for method, pattern in ORIGINAL_RUNS.items():
        for seed in OLD_SEEDS:
            run = R / pattern.format(seed=seed)
            config = read(run / 'config.json')
            summary = read(run / 'summary.json')
            assert summary['status'] == 'success' and summary['optimizer_steps'] == 512
            assert config['schedule'] == base.schedule(ids, seed, 512) == new_schedule(ids, seed, 512)
            assert schedule_v20(REPLAY_STEPS)(ids, seed, 512) == config['schedule'][:REPLAY_STEPS]
            for name, h in config['source_hashes'].items():
                assert sha(R / name) == h, name
            steps = rows(run / 'steps.jsonl')
            assert len(steps) == 512 and [s['source_id'] for s in steps] == config['schedule']
            caches.add(summary['cache_sha256'])
            seeds['original'][f'{method}-{seed}'] = dict(
                initial_reader_identity=summary['initial_reader_identity'],
                final_reader_identity=summary['final_reader_identity'],
                target_joint=summary['evaluations']['512']['probe']['counts']['target_joint'],
                steps_sha256=sha(run / 'steps.jsonl'), summary_sha256=sha(run / 'summary.json'))
    assert len(caches) == 1
    seeds['cache_sha256'] = caches.pop()
    for seed in OLD_SEEDS:
        assert seeds['original'][f'sft-{seed}']['initial_reader_identity'] == \
            seeds['original'][f'bottleneck-{seed}']['initial_reader_identity']
    import hashlib
    for seed in NEW_SEEDS:
        order = new_schedule(ids, seed, 512)
        assert len(order) == 512 == len(set(order))
        seeds['new_schedule_sha256'][str(seed)] = hashlib.sha256(json.dumps(order).encode()).hexdigest()
        try:
            base.schedule(ids, seed, 512)
            raise AssertionError('The original registry was expected to refuse the new seed')
        except ValueError:
            pass
    result = dict(
        status='pass', frozen_before_new_inference=True, lock_sha256=None if dry else sha(LOCK),
        original_protocol_sha256=sha(original_path), v19_prospective_protocol_sha256=sha(v19_path),
        data_sha256=data, cohort_shape=shape, choice=dict(jobs=jobs, references=references), seeds=seeds,
        time=time.time(), reserved_test_accessed=False)
    if lock is not None:
        assert lock['original_protocol_sha256'] == result['original_protocol_sha256']
        assert lock['v19_prospective_protocol_sha256'] == result['v19_prospective_protocol_sha256']
        assert lock['data_sha256'] == data
        write(O / 'preflight.json', result)
    summary = dict(preflight='pass', dry=dry, data=data, shape=shape, cache=seeds['cache_sha256'][:12],
                   references={k: {n: v['correct'] for n, v in c.items()} for k, c in references.items()})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main('--dry' in sys.argv[1:])
