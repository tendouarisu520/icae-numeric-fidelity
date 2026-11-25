"""Post hoc audit of recorded generations; no inference, tuning or test access."""
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'code/src'))
from binding.v11_pipeline import em_f1
from binding.v11_training import schedule
from binding.v13_protocol import metrics

def load(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]

def save(name, rows):
    with (P / 'tables' / name).open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

queries = ['first_team_first_mode_badge', 'first_team_second_mode_badge',
           'first_team_first_mode_locker', 'second_team_first_mode_badge']
contexts = ['original', 'swap', 'irrelevant_edit', 'reverse_order']
curves, question_rows, transitions, joints, pins = [], [], [], [], []
train = load(P / 'data/v13-answerable-relations-20261026/train.jsonl')
ids = list(dict.fromkeys(u['source_id'] for u in train))
assert len(ids) == 512 and Counter(Counter(u['source_id'] for u in train).values()) == {2: 512}
exposures = []
for seed in (73, 97):
    order = schedule(ids, seed, 512)
    assert len(order) == len(set(order)) == 512 and set(order) == set(ids)
    exposures.append(dict(seed=seed,updates=512,sources=512,min_exposure=1,max_exposure=1,
        evidence='reconstructed from pinned deterministic schedule and frozen training source order'))

files = [(p, 'development') for p in sorted((P/'evidence/recorded/runs').glob('*/main-*/evaluation-*/predictions.jsonl'))]
files += [(p, 'confirmation') for p in sorted((P/'confirmation').glob('main-*/predictions.jsonl'))]
for path, split in files:
    name = path.parent.parent.name if split == 'development' else path.parent.name
    _, arm, method, seed_string = name.split('-')
    seed = int(seed_string.removeprefix('seed'))
    step = int(path.parent.name.split('-')[1]) if split == 'development' else 512
    meta = dict(split=split,arm=arm,method=method,seed=seed,step=step)
    raw = load(path)
    pins.append(dict(path=str(path.relative_to(P)).replace('\\','/'),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    rows = [r for r in raw if r['family'] == 'probe']
    groups = defaultdict(dict)
    for r in rows:
        key = r['context_index'], r['question_index']
        assert key not in groups[r['unit_id']]
        groups[r['unit_id']][key] = int(em_f1(r['prediction'], r['gold'])[0])
    assert len(groups) == (96 if split == 'development' else 192)
    assert all(set(g) == {(c,q) for c in range(4) for q in range(4)} for g in groups.values())
    n = len(groups)
    target = sum(all(g[c,q] for c in (0,1) for q in (0,1)) for g in groups.values())
    unaffected = sum(all(g[c,q] for c in range(4) for q in (2,3)) for g in groups.values())
    m = metrics(rows)
    assert m['counts']['target_joint'] == [target,n]
    assert m['counts']['locality_joint'] == [unaffected,n]
    if split == 'development':
        curves.append(dict(**meta,target_successes=target,groups=n,
            unaffected_successes=unaffected,controlled_correct=sum(sum(g.values()) for g in groups.values()),controlled_rows=len(rows)))
    if step != 512:
        continue
    for q in range(4):
        for c, context in list(enumerate(contexts)) + [(-1, 'all_context_rows')]:
            cs = range(4) if c == -1 else [c]
            question_rows.append(dict(**meta,question_index=q,query=queries[q],context=context,
                correct=sum(g[i,q] for g in groups.values() for i in cs),total=n*len(cs)))
        joints.append(dict(**meta,question_index=q,query=queries[q],
            original_correct=sum(g[0,q] for g in groups.values()),
            all_four_correct=sum(all(g[c,q] for c in range(4)) for g in groups.values()),groups=n))
    for q in (2,3):
        for c in (1,2,3):
            counts = Counter((g[0,q],g[c,q]) for g in groups.values())
            transitions.append(dict(**meta,question_index=q,query=queries[q],intervention=contexts[c],
                correct_to_correct=counts[1,1],correct_to_wrong=counts[1,0],
                wrong_to_correct=counts[0,1],wrong_to_wrong=counts[0,0],
                original_correct=counts[1,1]+counts[1,0],original_wrong=counts[0,1]+counts[0,0],groups=n))
        base = sum(g[0,q] for g in groups.values())
        stay = sum(all(g[c,q] for c in range(4)) for g in groups.values())
        transitions.append(dict(**meta,question_index=q,query=queries[q],intervention='any_of_three',
            correct_to_correct=stay,correct_to_wrong=base-stay,
            wrong_to_correct='',wrong_to_wrong='',original_correct=base,original_wrong=n-base,groups=n))
    joints.append(dict(**meta,question_index='2+3',query='both_unaffected_queries',
        original_correct=sum(g[0,2] and g[0,3] for g in groups.values()),all_four_correct=unaffected,groups=n))

assert len(pins)==40 and len(curves)==30
save('learning_curve_counts.csv',curves)
save('query_context_accuracy.csv',question_rows)
save('query_joint_correctness.csv',joints)
save('unaffected_transitions.csv',transitions)
audit=dict(date='2026-11-06',status='post_hoc_descriptive_audit',
    training_exposure=exposures,original_config_files_available=False,
    convergence_demonstrated=False,forced_choice_inference_performed=False,
    reserved_test_accessed=False,model_inference_performed=False,
    query_roles_fixed=True,field_entity_and_query_position_confounded=True,
    prediction_files=pins)
(P/'reproduction/interpretation_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
for seed in (73,97):
    print('SEED',seed)
    for r in curves:
        if r['method']=='sft' and r['seed']==seed:
            print('CURVE',r)
    for collection,tag in ((question_rows,'QUESTION'),(joints,'JOINT'),(transitions,'TRANSITION')):
        for r in collection:
            if r['split']=='confirmation' and r['arm']=='compressed' and r['method']=='sft' and r['seed']==seed:
                if tag!='QUESTION' or r['context'] in ('original','all_context_rows'):
                    print(tag,r)
print(json.dumps(dict(prediction_files=40,learning_curve_cells=30,query_rows=len(question_rows),transition_rows=len(transitions))))
