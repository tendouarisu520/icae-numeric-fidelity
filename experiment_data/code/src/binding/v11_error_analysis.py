"""Read-only matched development error analysis; no model/data relabeling."""
from collections import Counter
import json
from pathlib import Path
from .v11_pipeline import em_f1,metrics,normalize,read_rows,sha,write_json


def indexed(rows):
    keys=[(r['unit_id'],r['context_index'],r['question_index']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate prediction key')
    return dict(zip(keys,rows))


def slice_stats(rows):
    known=[r for r in rows if r['gold']!='UNKNOWN']
    unknown=[r for r in rows if r['gold']=='UNKNOWN']
    answered=[r for r in known if normalize(r['prediction'])!='unknown']
    correct=lambda r:em_f1(r['prediction'],r['gold'])[0]
    return dict(counts=dict(exact=[sum(map(correct,rows)),len(rows)],
        known_exact=[sum(map(correct,known)),len(known)],unknown_exact=[sum(map(correct,unknown)),len(unknown)],
        known_false_unknown=[sum(normalize(r['prediction'])=='unknown' for r in known),len(known)]),
        known_f1=sum(em_f1(r['prediction'],r['gold'])[1] for r in known)/len(known) if known else None,
        wrong_when_answering=[sum(not correct(r) for r in answered),len(answered)])


def describe(rows):
    indexed(rows);result=metrics(rows);extra=slice_stats(rows)
    result.update(known_f1=extra['known_f1'],wrong_when_answering=extra['wrong_when_answering'],
        generation_without_eos=[sum(not r.get('generation',{}).get('eos_emitted',False) for r in rows),len(rows)],
        by_question={name:slice_stats([r for r in rows if r['question_index']==j])
                     for j,name in enumerate(('target','unaffected','genuine_unknown'))},
        by_context={name:slice_stats([r for r in rows if r['context_index']==i])
                     for i,name in enumerate(('original','answer_edit','support_deleted','irrelevant_edit'))})
    return result


def compare(baseline,candidate):
    left,right=indexed(baseline),indexed(candidate)
    if set(left)!=set(right):raise ValueError('Prediction groups differ')
    # Complete groups are required for every paired comparison.
    metrics(baseline);metrics(candidate)
    result=Counter(known_rescued=0,known_regressed=0,false_unknown_rescued=0,false_unknown_introduced=0,
                   unknown_rescued=0,unknown_regressed=0,candidate_correct_baseline_refused=0,refusal_to_wrong_answer=0)
    examples={name:[] for name in ('known_rescued','known_regressed','false_unknown_rescued','false_unknown_introduced')}
    known=0
    for key,a in sorted(left.items()):
        b=right[key]
        if a['gold']!=b['gold'] or a['candidates']!=b['candidates']:raise ValueError('Gold/candidate alignment differs')
        ca,cb=(bool(em_f1(r['prediction'],r['gold'])[0]) for r in (a,b))
        changes=[]
        if a['gold']=='UNKNOWN':
            if cb and not ca:result['unknown_rescued']+=1
            if ca and not cb:result['unknown_regressed']+=1
        else:
            known+=1
            ua,ub=(normalize(r['prediction'])=='unknown' for r in (a,b))
            if cb and not ca:changes.append('known_rescued')
            if ca and not cb:changes.append('known_regressed')
            if ua and not ub:changes.append('false_unknown_rescued')
            if not ua and ub:changes.append('false_unknown_introduced')
            if ua and cb:result['candidate_correct_baseline_refused']+=1
            if ua and not ub and not cb:result['refusal_to_wrong_answer']+=1
        for name in changes:
            result[name]+=1
            examples[name].append(dict(unit_id=key[0],context_index=key[1],question_index=key[2],
                gold=a['gold'],baseline=a['prediction'],candidate=b['prediction']))
    result=dict(result)
    result.update(known_rows=known,known_accuracy_delta=(result['known_rescued']-result['known_regressed'])/known,
                  examples=examples)
    return result


def collect(root):
    root=Path(root);runs=root/'runs/v11-natural-20261025';cells={};comparisons={}
    for seed in (73,97):
        entries={}
        for arm,method in (('encoder','sft'),('encoder','relation'),('full','sft')):
            name=f'main-{arm}-{method}-seed{seed}';folder=runs/name
            summary=json.loads((folder/'summary.json').read_text(encoding='utf8'))
            result=summary['evaluations']['1024']['development'];path=folder/'evaluation-1024/development/predictions.jsonl'
            if summary['status']!='success' or summary['optimizer_steps']!=1024 or sha(path)!=result['predictions_sha256']:
                raise ValueError('Incomplete or changed1024 result:'+name)
            entries[(arm,method)]=read_rows(path)
            cells[name]=describe(entries[(arm,method)])
            cells[name]['predictions_sha256']=sha(path)
        comparisons[str(seed)]={
            'relation_vs_sft':compare(entries[('encoder','sft')],entries[('encoder','relation')]),
            'full_vs_compressed_sft':compare(entries[('encoder','sft')],entries[('full','sft')]),
            'full_vs_compressed_relation':compare(entries[('encoder','relation')],entries[('full','sft')])}
    priorities=[]
    if all(comparisons[str(s)]['full_vs_compressed_sft']['known_accuracy_delta']>=.05 for s in (73,97)):
        priorities.append('Freeze each final compressor and train a matched rank16 reader on unchanged memories; same512 train units, seeds73/97, <=1024updates. This supplies the missing representation/reader crossing control.')
    if any(cells[f'main-{a}-{m}-seed{s}']['rates']['known_false_unknown']>=.10
           for s in (73,97) for a,m in (('encoder','sft'),('encoder','relation'),('full','sft'))):
        priorities.append('Run UNKNOWN-vs-answer score and forced-non-UNKNOWN recoverability diagnostics on development; keep raw greedy metrics primary. Any refusal threshold is selected only on development and then frozen.')
    priorities.append('Choose further changes from paired raw errors and both seeds; do not change confirmation/test or reinterpret forced answers as mixed-task accuracy.')
    return dict(status='complete',endpoint=1024,cells=cells,comparisons=comparisons,next_diagnostics=priorities,
                selection_partition='development',confirmation_evaluated=False,reserved_test_accessed=False)


def report(root,output):
    output=Path(output);output.mkdir(exist_ok=False)
    value=collect(root);write_json(output/'analysis.json',value)
    lines=['# V11 matched development errors','',
        '| Cell | Known correct | False UNKNOWN | Wrong / answered known | Target all four | Genuine UNKNOWN | Deleted target UNKNOWN |',
        '|---|---|---|---|---|---|---|']
    for name,c in value['cells'].items():
        f=lambda key:'/'.join(map(str,c['counts'][key]))
        wrong='/'.join(map(str,c['wrong_when_answering']))
        lines.append(f"|{name}|{f('known_exact')}|{f('known_false_unknown')}|{wrong}|{f('target_all_four')}|{f('genuine_unknown_exact')}|{f('deleted_target_unknown_exact')}|")
    lines+=['','Next diagnostic priorities:','']+['- '+x for x in value['next_diagnostics']]
    (output/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return value
