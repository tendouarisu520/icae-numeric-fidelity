"""V20 GPU evaluation with frozen readers: no optimizer, no model selection, no reserved test partition.

Modes
  smoke           two training-split units and one reader; exercises generation and candidate scoring
                  before any study inference
  crossed_choice  part A: candidate scoring by the four 512-update SFT readers on the two crossed-view cohorts;
                  generation is the stored greedy output of the earlier crossed-view evaluations
  new_seeds       part C: readouts of the three new compressed SFT readers on the four cohorts

Every cell writes rows.jsonl (all sixteen answers of each unit; candidate scores on the two target questions)
and receipt.json. Scoring rule, unchanged from the earlier choice study: the two candidates are the two target
values of the unit in the current context, tokenised with leading blank and EOS; the score is the mean token
log probability; token lengths must be equal; an exact tie is an error.
"""
from common import *
import gc
import sys

VIEW = ('semantic_rotation', 'layout_rotation', 'target_team_position')
OWNED = []  # output folders created by this process; a failure never rewrites the status of an earlier run


def identity_view(u):
    return all(u.get(k, 0) == 0 for k in VIEW)


def build_plan(mode, pre):
    """Return (jobs, cohorts). Each job lists its cells; a cell names a cohort and, optionally, stored generations."""
    cohorts = {}

    def cohort(name, limit=None, probe_only=False, path=None):
        if name not in cohorts:
            path = path or COHORTS[name]
            units = rows(R / path)
            if probe_only:
                units = [u for u in units if u['family'] == 'probe']
            if limit:
                units = units[:limit]
            cohorts[name] = dict(path=path, sha256=sha(R / path), units=units)
        return name

    jobs = []
    old = {j['label']: j for j in pre['choice']['jobs']}
    if mode == 'smoke':
        # Two training units (generation and scoring) and two development units of the original layout with the
        # stored outputs of the original seed-73 run (replay and scoring). No crossed or new-scene unit is touched.
        j = old['main-compressed-sft-seed73-step512']
        stored = R / ORIGINAL_RUNS['sft'].format(seed=73) / 'evaluation-512/predictions.jsonl'
        jobs.append(dict(j, cells=[
            dict(cohort=cohort('smoke_training', 2, True, 'data/v13-answerable-relations-20261026/train.jsonl'),
                 reference=None),
            dict(cohort=cohort('smoke_development', 2, True, COHORTS['development_original']),
                 reference=dict(path=stored.relative_to(R).as_posix(), sha256=sha(stored)))]))
    elif mode == 'crossed_choice':
        for label in SFT_LABELS:
            cells = [dict(cohort=cohort(name), reference=pre['choice']['references'][label][name])
                     for name in ('development_crossed', 'new_scene_crossed')]
            jobs.append(dict(old[label], cells=cells))
    elif mode == 'new_seeds':
        for seed in NEW_SEEDS:
            run = RUNS / f'main-compressed-sft-seed{seed}'
            summary = read(run / 'summary.json')
            assert summary['status'] == 'success' and summary['optimizer_steps'] == 512 and summary['seed'] == seed
            checkpoint = run / 'final-adapter.safetensors'
            assert sha(checkpoint) == summary['final_checkpoint_sha256']
            stored = run / 'evaluation-512/predictions.jsonl'
            assert sha(stored) == summary['evaluations']['512']['predictions_sha256']
            cells = [dict(cohort=cohort('development_original', probe_only=True),
                          reference=dict(path=stored.relative_to(R).as_posix(), sha256=sha(stored)))]
            cells += [dict(cohort=cohort(name), reference=None)
                      for name in ('new_scene_original', 'development_crossed', 'new_scene_crossed')]
            jobs.append(dict(name=f'main-compressed-sft-seed{seed}', label=f'main-compressed-sft-seed{seed}-step512',
                             arm='compressed', seed=seed, method='sft', step=512,
                             checkpoint=checkpoint.relative_to(R).as_posix(),
                             checkpoint_sha256=summary['final_checkpoint_sha256'],
                             identity=summary['final_reader_identity'], cells=cells))
    else:
        raise ValueError('Unknown mode')
    for name, expected in pre['data_sha256'].items():
        if name in cohorts:
            assert cohorts[name]['sha256'] == expected, name
    return jobs, cohorts


def main(mode):
    environment()
    verify_lock()
    pre = read(O / 'preflight.json')
    assert pre['status'] == 'pass' and pre['frozen_before_new_inference'] and pre['lock_sha256'] == sha(LOCK)
    from binding.repro_probe import configure_environment, configure_torch
    configure_environment('det_auto')
    configure_torch('det_auto')
    import torch
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    from binding.model import load_icae
    from binding.v11_pipeline import em_f1
    from binding.v11_training import load_model
    from binding.v12_reader import build_ram_cache, greedy
    from binding.v13_data import prepare
    from binding.v13_protocol import prediction_rows, metrics
    from binding.v13_objectives import prefix, logits
    from binding.v9_model import load_state, adapter_state
    from binding.v4_model import state_identity
    from binding.v16_code_fidelity import choice_decision
    assert torch.cuda.mem_get_info()[0] >= 23 * 1024 ** 3, 'GPU occupied; never stop another task'
    dest = O / mode
    dest.mkdir(exist_ok=False)
    OWNED.append(dest)
    status = dest / 'status.json'
    jobs, cohorts = build_plan(mode, pre)
    for job in jobs:
        assert sha(R / job['checkpoint']) == job['checkpoint_sha256'], job['label']
    protocol = dict(
        mode=mode, lock_sha256=sha(LOCK), preflight_sha256=sha(O / 'preflight.json'),
        cohorts={k: dict(path=v['path'], sha256=v['sha256'], units=len(v['units'])) for k, v in cohorts.items()},
        jobs=jobs,
        scoring='mean token log probability of each of the two target values with leading blank and EOS; '
                'equal token lengths required; exact ties are errors; candidates are not shown in the prompt',
        generation='stored greedy outputs where a reference is listed, otherwise 32-token greedy decoding as in '
                   'the original evaluation',
        started_before_inference=time.time(), model_selection=False, reserved_test_accessed=False)
    write(dest / 'protocol.json', protocol)
    tokenizer = AutoTokenizer.from_pretrained(str(R / 'models/mistral'), local_files_only=True, use_fast=False)
    items = {k: {u['id']: prepare(u, tokenizer) for u in v['units']} for k, v in cohorts.items()}
    emit(status, phase='encoding', mode=mode)
    encoder, _ = load_icae(str(R / 'models/mistral'), str(R / 'models/icae/mistral_7b_ft_icae.safetensors'),
                           device='cuda', dtype=torch.bfloat16)
    encoder.requires_grad_(False)
    encoder.eval()
    versions = {n: p._version for n, p in encoder.named_parameters()}
    cache = build_ram_cache(encoder, [c for sp in items.values() for it in sp.values() for c in it['contexts']],
                            lambda **v: emit(status, phase='encoding', mode=mode, **v))
    assert versions == {n: p._version for n, p in encoder.named_parameters()}
    cache_hash = cache.digest()
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    model, _ = load_model('full', 73)
    model.eval()
    assert torch.equal(cache.ft, model.ft.detach().cpu())
    frozen = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    cells_done = 0
    for job in jobs:
        load_state(model, load_file(str(R / job['checkpoint']), device='cpu'))
        identity = state_identity(adapter_state(model))['sha256']
        assert identity == job['identity'], 'Reader identity differs from the pinned value'
        for cell in job['cells']:
            name = cell['cohort']
            units = cohorts[name]['units']
            reference = None
            if cell['reference'] is not None:
                path = R / cell['reference']['path']
                assert sha(path) == cell['reference']['sha256']
                reference = {(r['unit_id'], r['context_index'], r['question_index']): r for r in rows(path)}
            out = dest / job['label'] / name
            out.mkdir(parents=True, exist_ok=False)
            replayed = 0
            if reference is not None:
                first = next(u for u in units if identity_view(u))
                other = next((u for u in units if not identity_view(u)), None) or units[1]
                with torch.no_grad():
                    for u in (first, other):
                        item = items[name][u['id']]
                        for i in range(4):
                            for j in (0, 1):
                                pred, g = greedy(model, prefix(model, item, i, j, cache, job['arm']), tokenizer)
                                stored = reference[u['id'], i, j]
                                assert pred == stored['prediction'] and g == stored['generation'], \
                                    'Stored greedy output is not reproduced'
                                replayed += 1
            n = correct = free = ties = 0
            allrows = []
            start = time.perf_counter()
            with torch.no_grad(), (out / 'rows.jsonl').open('x', encoding='utf8', buffering=1) as f:
                for number, u in enumerate(units, 1):
                    item = items[name][u['id']]
                    predictions = []
                    extras = {}
                    for i in range(4):
                        values = []
                        for j in range(4):
                            p = prefix(model, item, i, j, cache, job['arm'])
                            if reference is None:
                                pred, g = greedy(model, p, tokenizer)
                            else:
                                stored = reference[u['id'], i, j]
                                assert stored['gold'] == u['answers'][i][j]
                                pred, g = stored['prediction'], stored['generation']
                            values.append(pred)
                            extra = dict(generation=g)
                            if j < 2:
                                candidates = item['answers'][i][:2]
                                assert len(candidates[0]) == len(candidates[1])
                                scores = []
                                for answer in candidates:
                                    output = logits(model, p, answer)
                                    truth = torch.tensor(answer, device=model.device)
                                    lp = output.log_softmax(-1).gather(-1, truth[:, None]).squeeze(-1)
                                    assert bool(torch.isfinite(lp).all())
                                    scores.append(dict(mean=float(lp.mean()), sum=float(lp.sum()),
                                                       token_logp=lp.tolist(), token_ids=answer))
                                choice = choice_decision([s['mean'] for s in scores])
                                ok = bool(em_f1(pred, u['answers'][i][j])[0])
                                extra.update(candidate_values=u['answers'][i][:2], candidate_scores=scores,
                                             choice=choice, choice_correct=choice == j, greedy_correct=ok,
                                             gold_margin=scores[j]['mean'] - scores[1 - j]['mean'])
                                n += 1
                                correct += choice == j
                                free += ok
                                ties += choice is None
                            extras[i, j] = extra
                        predictions.append(values)
                    group = prediction_rows(u, predictions)
                    for row in group:
                        row.update(extras[row['context_index'], row['question_index']])
                        f.write(json.dumps(row, allow_nan=False) + '\n')
                    allrows.extend(group)
                    if number == 1 or number % 8 == 0 or number == len(units):
                        emit(status, phase='evaluation', mode=mode, job=job['label'], cohort=name, units=number,
                             total_units=len(units), cells_done=cells_done, seconds=time.perf_counter() - start)
            receipt = dict(job=job['label'], cohort=name, metrics=metrics(allrows), rows=len(allrows), target_rows=n,
                           choice_correct=correct, greedy_correct=free, ties=ties,
                           rows_sha256=sha(out / 'rows.jsonl'), protocol_sha256=sha(dest / 'protocol.json'),
                           reader_identity=identity, checkpoint_sha256=job['checkpoint_sha256'],
                           generated=reference is None, replay_rows=replayed,
                           reference=cell['reference'], wall_seconds=time.perf_counter() - start)
            write(out / 'receipt.json', receipt)
            cells_done += 1
        assert identity == state_identity(adapter_state(model))['sha256']
    assert frozen == {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    assert cache.digest() == cache_hash
    emit(status, phase='complete', mode=mode, jobs=len(jobs), cells=cells_done)


if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except BaseException as e:
        if OWNED:
            try:
                emit(OWNED[0] / 'status.json', phase='failed', error=repr(e))
            except Exception:
                pass
        raise
