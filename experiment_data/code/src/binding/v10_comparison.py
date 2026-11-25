"""Collect all coefficient cells at1024, including unchanged historical controls."""
import json
from pathlib import Path
from .v11_pipeline import sha,write_json


def compact(value):
    k,u=value['known_rows'],value['unknown_rows'];pairs=k//2
    return dict(known=[value['known_correct'],k],unknown=[value['unknown_correct'],u],
                pair=[round(value['pair_accuracy']*pairs),pairs],
                misbinding=[round(value['foil_misbinding_rate']*k),k],
                false_unknown=[round(value['false_unknown_rate']*k),k])


def collect(root):
    root=Path(root);cells=[]
    for seed in (73,97):
        for method in ('sft','hn','paired','row05','row10col05'):
            old=method in ('sft','hn','paired')
            path=root/(f'runs/v9-pretrained-20261024/encoder-{method}-seed{seed}/summary.json' if old else f'runs/v10-coefficient-20261025/{method}-seed{seed}/summary.json')
            result=json.loads(path.read_text(encoding='utf8'))
            if result['status']!='success' or result['optimizer_steps']!=1024:raise ValueError('Incomplete cell:'+str(path))
            evaluations=result['evaluations']['1024'] if old else result['evaluations']
            cell=dict(method=method,seed=seed,summary_sha256=sha(path),
                      training_seconds=result['cumulative_training_seconds'],splits={})
            for split in ('train_d12','val_id','val_ood'):
                cell['splits'][split]=dict(pooled=compact(evaluations[split]['pooled']),d12=compact(evaluations[split]['by_level']['d12']))
            cells.append(cell)
    index={(c['method'],c['seed']):c for c in cells};comparisons={}
    rate=lambda x:x[0]/x[1]
    for method in ('row05','row10col05'):
        comparisons[method]={}
        for seed in (73,97):
            value=index[(method,seed)]['splits']['val_id'];base=index[('hn',seed)]['splits']['val_id']
            delta=rate(value['d12']['pair'])-rate(base['d12']['pair'])
            unknown=rate(value['pooled']['unknown'])-rate(base['pooled']['unknown'])
            comparisons[method][str(seed)]=dict(d12_pair_delta=delta,pooled_unknown_delta=unknown,target_met=delta>=.05-1e-9 and unknown>=-.02-1e-9)
        comparisons[method]['both_seeds_target_met']=all(comparisons[method][str(s)]['target_met'] for s in (73,97))
    output=root/'results/v10-coefficient-comparison-20261025';output.mkdir(exist_ok=False)
    report=dict(status='complete',endpoint=1024,cells=cells,comparisons=comparisons,confirmation_used=False,reserved_test_accessed=False)
    write_json(output/'comparison.json',report)
    lines=['# V10 complete coefficient comparison','', 'All rows use the final1024 endpoint.','',
           '| Seed | Method | ID d12 pair | ID pooled pair | ID UNKNOWN | ID misbinding |',
           '|---|---|---|---|---|---|']
    for c in cells:
        v=c['splits']['val_id'];fmt=lambda a:f'{a[0]}/{a[1]}'
        lines.append(f"|{c['seed']}|{c['method']}|{fmt(v['d12']['pair'])}|{fmt(v['pooled']['pair'])}|{fmt(v['pooled']['unknown'])}|{fmt(v['pooled']['misbinding'])}|")
    (output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return report
