"""Freeze the plan, code, existing cohort and new scene bytes before GPU use."""
from pathlib import Path
import ast,hashlib,json,sys,zipfile
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'src'))
from binding.v19_data import prospective_units,balanced_training
P=R/'deliverables/20261106/Numeric_Fidelity_Reviewed_v3_20261106'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
out=R/'data/v19-prospective-20261108';out.mkdir(exist_ok=False)
ordinary,crossed=prospective_units();balanced=balanced_training([u for u in rows(P/'data/v13-answerable-relations-20261026/train.jsonl') if u['family']=='probe'])
pins={}
for name,units in [('original',ordinary),('crossed',crossed),('balanced_training',balanced)]:
 p=out/(name+'.jsonl');p.write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in units),encoding='utf8',newline='');pins[name]=sha(p)
oldfiles=[P/'data/v13-answerable-relations-20261026'/f'{s}.jsonl' for s in ('train','development')]+[P/'confirmation/confirmation_units.jsonl',R/'data/v16-role-balance-20261106/development.jsonl']
old=[u for p in oldfiles for u in rows(p)]
assert not {u['source_id'] for u in ordinary}&{u['source_id'] for u in old}
assert not {c for u in ordinary+crossed for c in u['contexts']}&{c for u in old for c in u['contexts']}
new=[p for p in (R/'temp/v19').glob('*.py')]+[R/f'src/binding/v19_{n}.py' for n in ('training','data','analysis')]+[R/'src/tests/test_v19_followup.py',R/'docs/superpowers/plans/2026-11-08-budget-layout-prospective.md']
new=[p for p in new if p.name not in ('freeze.py','build_training.py')]
for p in new:
 if p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'),filename=str(p))
oldmodules=[R/'src/binding'/n for n in ('v13_data.py','v13_protocol.py','v13_objectives.py','v14_objectives.py','v13_training.py','v16_training.py','v16_exposure.py','v16_roles.py','v16_code_fidelity.py','v12_reader.py','v11_training.py','v11_pipeline.py','v4_model.py','v9_model.py','model.py','benchmark.py','repro_probe.py')]
for p in oldmodules:
 assert sha(p)==sha(P/'code/src/binding'/p.name),p
lock=dict(date='2026-11-08',status='fixed_before_gpu',files={p.relative_to(R).as_posix():sha(p) for p in new+oldmodules},generated_data_sha256=pins,original_protocol_sha256=sha(P/'confirmation/protocol.json'),test_accessed=False,plan='docs/superpowers/plans/2026-11-08-budget-layout-prospective.md')
path=R/'temp/v19/lock.json';assert not path.exists();path.write_text(json.dumps(lock,indent=2)+'\n',encoding='utf8')
with zipfile.ZipFile(R/'temp/v19/upload.zip','x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in new+[path]:z.write(p,p.relative_to(R).as_posix())
receipt=dict(sha256=sha(R/'temp/v19/upload.zip'),bytes=(R/'temp/v19/upload.zip').stat().st_size,files=len(new)+1,lock_sha256=sha(path))
(R/'temp/v19/upload_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
