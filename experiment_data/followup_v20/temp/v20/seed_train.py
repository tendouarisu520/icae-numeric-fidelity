"""V20 part B entry point: the unchanged v13/v14 train() at seeds fixed in the V20 plan.

python seed_train.py main   <sft|bottleneck> <101|131|151>   one 512-update run, development evaluation as coded
python seed_train.py replay <sft|bottleneck> 73              first 32 updates of the original seed-73 run

Nothing in src/binding is edited. Two module attributes are replaced in this process only:
  RUNS of the training module       new output folder (runs/v20-seed-extension-20261113)
  binding.v11_training.schedule     same algorithm; the registered seeds are 73, 97, 101, 131, 151
In replay mode the order is cut to its first 32 entries, so the learning-rate schedule (computed for 512
updates) and every other setting are those of the original run; the per-update log is then compared with the
original log.
"""
from common import *
import argparse
import math
import sys


def compare_replay(method, summary):
    name = f'main-compressed-{method}-seed73'
    old_dir = R / ORIGINAL_RUNS[method].format(seed=73)
    new = rows(RUNS / 'replay32' / name / 'steps.jsonl')
    old = rows(old_dir / 'steps.jsonl')[:REPLAY_STEPS]
    old_summary = read(old_dir / 'summary.json')
    assert len(new) == len(old) == REPLAY_STEPS
    exact = all(a[k] == b[k] for a, b in zip(new, old)
                for k in ('step', 'source_id', 'loss', 'families', 'gradient_norm', 'learning_rate'))
    order = all(a[k] == b[k] for a, b in zip(new, old) for k in ('step', 'source_id', 'learning_rate'))
    rel = lambda x, y: abs(x - y) / max(abs(y), 1e-12)
    result = dict(
        method=method, seed=73, updates=REPLAY_STEPS, exact=exact, order_and_learning_rate_exact=order,
        max_abs_loss_difference=max(abs(a['loss'] - b['loss']) for a, b in zip(new, old)),
        max_relative_loss_difference=max(rel(a['loss'], b['loss']) for a, b in zip(new, old)),
        max_relative_gradient_norm_difference=max(rel(a['gradient_norm'], b['gradient_norm']) for a, b in zip(new, old)),
        initial_reader_identity_equal=summary['initial_reader_identity'] == old_summary['initial_reader_identity'],
        cache_sha256_equal=summary['cache_sha256'] == old_summary['cache_sha256'],
        original_steps_sha256=sha(old_dir / 'steps.jsonl'), replay_steps_sha256=sha(RUNS / 'replay32' / name / 'steps.jsonl'),
        first_loss=[new[0]['loss'], old[0]['loss']], last_loss=[new[-1]['loss'], old[-1]['loss']])
    # Fixed in the plan: identities, order and learning rate must be exact; losses and gradient norms are
    # expected to be exact, and a relative difference up to 1e-4 is recorded and tolerated.
    result['pass'] = bool(order and result['initial_reader_identity_equal'] and result['cache_sha256_equal']
                          and result['max_relative_loss_difference'] <= 1e-4
                          and result['max_relative_gradient_norm_difference'] <= 1e-4
                          and math.isfinite(result['max_abs_loss_difference']))
    write(RUNS / 'replay32' / f'replay_check-{method}.json', result)
    print(json.dumps(result), flush=True)
    if not result['pass']:
        raise RuntimeError('Replay of the original run differs; new seeds are not started')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('replay', 'main'))
    parser.add_argument('method', choices=('sft', 'bottleneck'))
    parser.add_argument('seed', type=int)
    a = parser.parse_args()
    environment()
    verify_lock()
    pre = read(O / 'preflight.json')
    assert pre['status'] == 'pass' and pre['lock_sha256'] == sha(LOCK)
    assert a.seed == 73 if a.mode == 'replay' else a.seed in NEW_SEEDS
    assert os.environ.get('PYTHONHASHSEED') == '73', 'Original process environment required'
    import binding.v11_training as base
    if a.method == 'sft':
        import binding.v13_training as module
    else:
        import binding.v14_training as module
    base.schedule = schedule_v20(REPLAY_STEPS if a.mode == 'replay' else None)
    module.RUNS = RUNS / 'replay32' if a.mode == 'replay' else RUNS
    module.RUNS.mkdir(parents=True, exist_ok=True)
    summary = module.train(argparse.Namespace(phase='main', arm='compressed', method=a.method, seed=a.seed))
    steps = REPLAY_STEPS if a.mode == 'replay' else 512
    assert summary['status'] == 'success' and summary['optimizer_steps'] == steps
    assert all(summary[k] for k in ('cache_unchanged', 'frozen_parameters_unchanged', 'reload_exact'))
    if a.mode == 'replay':
        compare_replay(a.method, summary)
    else:
        assert sorted(summary['evaluations']) == ['128', '256', '512']
        assert summary['cache_sha256'] == pre['seeds']['cache_sha256'], 'Frozen memories differ from the original runs'


if __name__ == '__main__':
    main()
