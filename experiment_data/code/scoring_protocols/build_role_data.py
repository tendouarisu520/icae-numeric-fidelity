from pathlib import Path
import hashlib
import json
import sys
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'src'))
from binding.v16_roles import counterbalanced_variants
P=R/'deliverables/20261106/Relation_Reading_Corrected'
path=P/'data/v13-answerable-relations-20261026/development.jsonl'
if not path.exists():path=R/'data/v13-answerable-relations-20261026/development.jsonl'
base=[json.loads(x) for x in path.read_text().splitlines()]
base=[u for u in base if u['family']=='probe' and u['level']==4]
selected=[];articles=set()
for u in sorted(base,key=lambda u:hashlib.sha256(('v16-role-source:'+u['source_id']).encode()).hexdigest()):
    if u['article'] in articles:continue
    selected.append(u);articles.add(u['article'])
    if len(selected)==8:break
assert len(selected)==len(articles)==8
units=[v for u in selected for v in counterbalanced_variants(u)]
out=R/'data/v16-role-balance-20261106';out.mkdir(parents=True,exist_ok=False)
data=''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in units).encode()
(out/'development.jsonl').write_bytes(data)
cfg=dict(date='2026-11-06',split='development',purpose='post hoc factorial diagnostic, not new independent confirmation',
    selection='first8 distinct development articles by sha256(v16-role-source:+source_id), level4 only',
    selected_sources=[dict(id=u['source_id'],article=u['article']) for u in selected],
    factors=dict(target_field_identity=3,within_record_field_position=3,first_two_team_role=2),
    original_contexts=4,queries=4,units=len(units),rows=16*len(units),
    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),data_sha256=hashlib.sha256(data).hexdigest(),
    generator_sha256=hashlib.sha256((R/'src/binding/v16_roles.py').read_bytes()).hexdigest(),
    comparisons=['original512compressedSFT73','original512compressedSFT97','original512fullSFT73','original512fullSFT97','extended1024compressedSFT73','extended1024compressedSFT97'],
    no_training_on_diagnostic=True,reserved_test_accessed=False)
(out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf8')
print(json.dumps(cfg))
