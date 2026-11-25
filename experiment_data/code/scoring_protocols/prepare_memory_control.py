from pathlib import Path
import hashlib,itertools,json
R=Path(__file__).resolve().parents[2]
P=R/'deliverables/20261106/Relation_Reading_Corrected'
data=P/'data/v13-answerable-relations-20261026/development.jsonl'
units=[json.loads(s) for s in data.read_text(encoding='utf8').splitlines()]
selection=json.loads((R/'data/v16-role-balance-20261106/protocol.json').read_text())['selected_sources']
units=[next(u for u in units if u['family']=='probe' and u['level']==4 and u['source_id']==s['id']) for s in selection]
assert len(units)==len({u['article'] for u in units})==8
valid=lambda perm:all(i!=j and not {a for row in units[i]['answers'] for a in row[:2]}&{f['value'] for facts in units[j]['facts'] for f in facts} for i,j in enumerate(perm))
cyclic=[tuple((i+k)%8 for i in range(8)) for k in range(1,8)]
donors=next((p for p in cyclic if valid(p)),None)
if donors is None:donors=next(p for p in itertools.permutations(range(8)) if valid(p))
plan=dict(date='2026-11-06',purpose='Matched-versus-shuffled frozen-memory conditional-choice diagnostic; no training or selection',
    selection='Same eight source articles as the factorial role diagnostic, original level4 views only',
    source_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),modes=['matched','shuffled'],
    donor_rule='First valid cyclic shift of the fixed source order; if none exists, first valid lexicographic derangement. Original target codes must be absent from all donor contexts.',
    pairs=[dict(unit_id=u['id'],source_id=u['source_id'],article=u['article'],donor_unit_id=units[j]['id'],donor_source_id=units[j]['source_id'],donor_article=units[j]['article']) for u,j in zip(units,donors)],
    expected_rows_per_reader=128,expected_matched_rows=64,readers=['main-compressed-sft-seed73','main-compressed-sft-seed97'],split='development',reserved_test_accessed=False)
dest=R/'temp/v16/memory_control_plan.json';assert not dest.exists();dest.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(path=str(dest),donor_indices=donors,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())))
