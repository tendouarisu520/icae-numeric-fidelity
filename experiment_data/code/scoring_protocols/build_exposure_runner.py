"""Derive a separately named trainer; leave all originally pinned files intact."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
R=Path(__file__).resolve().parents[2]
source=R/'src/binding/v13_training.py'
old=source.read_text(encoding='utf8')
s=old
def replace(a,b):
    global s
    assert s.count(a)==1,(a[:60],s.count(a))
    s=s.replace(a,b)
replace('"""Frozen-official-memory, answerable mixed QA, reader-only controlled study."""',
    '"""V16 exact original512 replay followed by512 additional SFT updates."""')
replace("RUNS=ROOT/'runs/v13-answerable-reading-20261026'","RUNS=ROOT/'runs/v16-repeated-exposure-20261106'\nfrom .v16_exposure import exposure_rate,checkpoints as main_checkpoints,plateau_diagnostic")
replace("    if ROOT.drive.lower()!='e:':raise ValueError('Remote E only')", "    if ROOT.drive.lower()!='e:':raise ValueError('Remote E only')\n    if args.phase!='main' or args.method!='sft':raise ValueError('V16 is main SFT exposure diagnostic only')\n    old_run=ROOT/'runs/v13-answerable-reading-20261026'/f'main-{args.arm}-sft-seed{args.seed}'\n    old_summary=json.loads((old_run/'summary.json').read_text(encoding='utf8'))\n    assert old_summary['status']=='success' and old_summary['optimizer_steps']==512")
replace("max_steps={'smoke':2,'overfit':1024,'main':512}[args.phase]","max_steps=1024")
replace("        initial=state_identity(adapter_state(model))['sha256']","        initial=state_identity(adapter_state(model))['sha256']\n        assert initial==old_summary['initial_reader_identity'],'Initialization differs from original reader'")
replace("            encoder_trainable=False,runtime=runtime_settings(),confirmation_loaded=False,reserved_test_accessed=False))", "            encoder_trainable=False,runtime=runtime_settings(),confirmation_loaded=False,reserved_test_accessed=False,\n            source_exposures=2,lr_extension='original512 schedule then constant5e-6',\n            old_summary_sha256=sha(old_run/'summary.json'),exposure_module_sha256=sha(ROOT/'src/binding/v16_exposure.py')))")
replace("rate=learning_rate(completed,max_steps,5e-5,32) if args.phase=='main' else 1e-4*min(completed/32,1.)","rate=exposure_rate(completed)")
replace("                checkpoints=(128,256,512,1024) if args.phase=='overfit' else (128,256,512)","                checkpoints=main_checkpoints\n                if completed==512:\n                    replay_identity=state_identity(adapter_state(model))['sha256']\n                    assert replay_identity==old_summary['final_reader_identity'],'Exact512adapterreplayfailed'\n                    save_file(adapter_state(model),str(output/'step512-adapter.safetensors'))")
replace("                    evaluations[str(completed)]=result;write_json(output/'evaluations.json',evaluations)","                    evaluations[str(completed)]=result;write_json(output/'evaluations.json',evaluations)\n                    if completed<=512:\n                        old_result=old_summary['evaluations'][str(completed)]\n                        assert result['predictions_sha256']==old_result['predictions_sha256'],'Original checkpoint generations differ'")
replace("        summary=dict(status='success' if passed else 'gate_failed'", "        steps=read_rows(output/'steps.jsonl')\n        losses=[.5*sum(v['sft_loss'] for v in x['families']) for x in steps[-256:]]\n        plateau=plateau_diagnostic([evaluations[str(k)]['probe']['counts']['target_joint'][0] for k in (768,896,1024)],losses)\n        summary=dict(source_exposures=2,step512_exact_replay=True,plateau_diagnostic=plateau,\n            status='success' if passed else 'gate_failed'")
ast.parse(s)
dest=R/'src/binding/v16_training.py'
dest.write_text(s,encoding='utf8',newline='\n')
(R/'temp/v16/exposure_diff.txt').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='pinned_v13_training.py',tofile='new_v16_training.py')),encoding='utf8')
print(json.dumps(dict(path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),original_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
