"""Minimal auditable extension of the already validated v16 training path."""
from pathlib import Path
import difflib
R=Path(__file__).resolve().parents[2];old=(R/'src/binding/v16_training.py').read_text(encoding='utf8');s=old
def change(a,b):
 global s
 assert s.count(a)==1,a;s=s.replace(a,b)
change('V16 exact original512 replay followed by512 additional SFT updates.','V19 matched auxiliary extension or exposure-balanced SFT, fixed1024 updates.')
change("RUNS=ROOT/'runs/v16-repeated-exposure-20261106'","RUNS=ROOT/'runs/v19-budget-layout-20261108'")
change('    from .v13_objectives import backward,prefix\n','    from .v13_objectives import prefix\n    if args.method==\'bottleneck\':\n        from .v14_objectives import backward\n    else:\n        from .v13_objectives import backward\n')
change("    if args.phase!='main' or args.method!='sft':raise ValueError('V16 is main SFT exposure diagnostic only')","    if args.phase!='main' or args.arm!='compressed':raise ValueError('V19 fixed main compressed experiment')\n    if args.balanced and args.method!='sft':raise ValueError('Balanced SFT only')\n    if not args.balanced and args.method not in ('assignment','relation','bottleneck'):raise ValueError('Three auxiliary extensions only')")
change("    old_run=ROOT/'runs/v13-answerable-reading-20261026'/f'main-{args.arm}-sft-seed{args.seed}'","    old_root='runs/v14-bottleneck-reading-20261026' if args.method=='bottleneck' else 'runs/v13-answerable-reading-20261026'\n    old_run=ROOT/old_root/f'main-{args.arm}-{args.method}-seed{args.seed}'")
change("    name=f'{args.phase}-{args.arm}-{args.method}-seed{args.seed}'","    name=f'{\"balanced\" if args.balanced else \"extended\"}-{args.arm}-{args.method}-seed{args.seed}'")
needle="        order=schedule(list(by_source),args.seed,max_steps)"
change(needle,needle+"\n        if args.balanced:\n            from .v19_data import balanced_training\n            natural=[u for u in train_units if u['family']=='natural']\n            augmented=balanced_training([u for u in train_units if u['family']=='probe'])\n            train_units=natural+augmented\n            by_source=defaultdict(list)\n            for u in train_units:by_source[u['source_id']].append(u)\n            assert all(len(v)==3 for v in by_source.values())")
change("        source_paths.extend(DATA/(s+'.jsonl') for s in splits)","        source_paths.extend(DATA/(s+'.jsonl') for s in splits)\n        source_paths.extend(ROOT/p for p in ('src/binding/v14_objectives.py','src/binding/v19_data.py','src/binding/v16_exposure.py','temp/v19/lock.json'))")
change("source_exposures=2,lr_extension='original512 schedule then constant5e-6',","source_exposures=2,balanced_layout=args.balanced,lr_extension='original512 schedule then constant5e-6',")
change("                values=[backward(model,items[u['id']],frozen_cache,args.arm,args.method,.5) for u in by_source[uid]]","                selected=[u for u in by_source[uid] if not args.balanced or u['family']=='natural' or u['exposure']==(completed-1)//512]\n                assert len(selected)==2\n                selected.sort(key=lambda u:u['family']!='natural')\n                values=[backward(model,items[u['id']],frozen_cache,args.arm,args.method,.5) for u in selected]")
change("                    assert replay_identity==old_summary['final_reader_identity'],'Exact512adapterreplayfailed'","                    if not args.balanced:assert replay_identity==old_summary['final_reader_identity'],'Exact512adapterreplayfailed'")
change('                    if completed<=512:','                    if completed<=512 and not args.balanced:')
change('summary=dict(source_exposures=2,step512_exact_replay=True,plateau_diagnostic=plateau,','summary=dict(source_exposures=2,balanced_layout=args.balanced,step512_exact_replay=None if args.balanced else True,plateau_diagnostic=plateau,')
change("        write_json(output/'summary.json',summary);update(status='success'","        torch.save(dict(step=completed,optimizer=optimizer.state_dict(),reader_identity=final_identity),output/'optimizer-final.pt')\n        write_json(output/'summary.json',summary);update(status='success'")
change("choices=('sft','assignment','relation'),default='sft'","choices=('sft','assignment','relation','bottleneck'),default='sft'")
change("default=73);args=parser.parse_args()","default=73);parser.add_argument('--balanced',action='store_true');args=parser.parse_args()")
out=R/'src/binding/v19_training.py';assert not out.exists();out.write_text(s,encoding='utf8')
(R/'temp/v19/training.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='v16_training.py',tofile='v19_training.py')),encoding='utf8')
print('Generated isolated training module; prior sources unchanged')
