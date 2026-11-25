"""Independently verify near-code controls and sequence likelihood comparisons."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,math,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code/src'))
from binding.v16_near_codes import near_code
O=R/'followup';F=O/'followups/near';P=R
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
units={'development':{u['id']:u for u in rows(P/'data/v13-answerable-relations-20261026/development.jsonl')},'confirmation':{u['id']:u for u in rows(P/'confirmation/confirmation_units.jsonl')}}
protocol=read(F/'protocol.json');summaries=[];strata=[];checks=[]
for job in protocol['jobs']:
    for split in job['splits']:
        folder=F/split/job['label'];receipt=read(folder/'receipt.json');p=folder/'scores.jsonl'
        assert receipt['exact_replay_rows']==2
        assert sha(p)==receipt['scores_sha256'] and sha(F/'protocol.json')==receipt['protocol_sha256']
        raw=rows(p);expected={(u['id'],i,j) for u in units[split].values() if u['family']=='probe' for i in range(4) for j in (0,1)}
        assert len(raw)==len(expected) and {(r['unit_id'],r['context_index'],r['question_index']) for r in raw}==expected
        if split=='confirmation':oldpath=P/'confirmation'/job['name']/'predictions.jsonl'
        elif job['step']==1024:oldpath=O/f"followups/exposure{job['seed']}/evaluation-1024/predictions.jsonl"
        else:oldpath=next((P/'evidence/recorded/runs').glob(f"*/{job['name']}/evaluation-512/predictions.jsonl"))
        assert sha(oldpath)==receipt['raw_predictions_sha256']
        old={(r['unit_id'],r['context_index'],r['question_index']):r for r in rows(oldpath)}
        prior_choice=None;maximum_replay_difference=0.0
        if job['step']==512:
            prior_choice={(r['unit_id'],r['context_index'],r['question_index']):r for r in rows(O/'choice'/split/job['name']/'scores.jsonl')}
        groups=defaultdict(dict)
        for r in raw:
            uid,i,j=r['unit_id'],r['context_index'],r['question_index'];u=units[split][uid];ref=old[uid,i,j]
            assert r['gold']==u['answers'][i][j]==ref['gold'] and r['prediction']==ref['prediction']
            assert r['source_id']==u['source_id'] and r['article']==u['article'] and r['level']==u['level']
            assert r['greedy_correct']==(r['prediction'].strip()==r['gold'])
            assert r['condition_foil_value']==u['answers'][i][1-j]
            for pos in (0,3):
                assert r['near'][str(pos)]['value']==near_code(r['gold'],[f['value'] for f in u['facts'][i]],f'{uid}:{i}:{j}',pos)
            scores=[r['gold_score'],r['condition_foil_score'],r['near']['0']['score'],r['near']['3']['score'],r['generated_score']]
            assert all(s['tokens']==6 for s in scores[:4]) and scores[-1]['tokens']==len(ref['generation']['token_ids'])
            for s in scores:
                assert len(s['token_logp'])==s['tokens'] and all(math.isfinite(v) and v<=1e-6 for v in s['token_logp'])
                assert abs(sum(s['token_logp'])-s['sum'])<1e-4 and abs(sum(s['token_logp'])/s['tokens']-s['mean'])<2e-5
            if prior_choice is not None:
                previous=prior_choice[uid,i,j]
                differences=[abs(scores[0]['mean']-previous['candidate_scores'][j]['mean']),abs(scores[1]['mean']-previous['candidate_scores'][1-j]['mean'])]
                maximum_replay_difference=max(maximum_replay_difference,*differences)
                assert max(differences)<=1e-5
                assert (scores[0]['mean']>scores[1]['mean'])==previous['choice_correct']
            for key,s in [('condition',scores[1]),('first_digit',scores[2]),('last_digit',scores[3])]:
                r[key+'_correct']=scores[0]['mean']>s['mean'];r[key+'_tie']=scores[0]['mean']==s['mean']
            r['gold_more_likely_wrong']=not r['greedy_correct'] and scores[0]['sum']>scores[4]['sum']
            r['sequence_logp_advantage']=scores[0]['sum']-scores[4]['sum']
            r['same_length']=scores[0]['tokens']==scores[4]['tokens'];groups[uid][i,j]=r
        calc=dict(rows=len(raw),condition_correct=sum(r['condition_correct'] for r in raw),first_digit_correct=sum(r['first_digit_correct'] for r in raw),last_digit_correct=sum(r['last_digit_correct'] for r in raw),
            greedy_correct=sum(r['greedy_correct'] for r in raw),wrong_generation_gold_more_likely=sum(r['gold_more_likely_wrong'] for r in raw))
        for k,v in calc.items():assert receipt[k]==v,(job['label'],k)
        meta=dict(split=split,arm=job['arm'],seed=job['seed'],step=job['step'])
        summaries.append(dict(**meta,**calc,wrong_generation=len(raw)-calc['greedy_correct'],same_length_wrong=sum(r['same_length'] and not r['greedy_correct'] for r in raw),
            same_length_gold_more_likely_wrong=sum(r['same_length'] and r['gold_more_likely_wrong'] for r in raw),
            same_length_wrong_gold_less_likely=sum(r['same_length'] and not r['greedy_correct'] and r['sequence_logp_advantage']<0 for r in raw),
            same_length_wrong_sequence_ties=sum(r['same_length'] and not r['greedy_correct'] and r['sequence_logp_advantage']==0 for r in raw),
            same_length_wrong_gold_advantage_gt0p01=sum(r['same_length'] and not r['greedy_correct'] and r['sequence_logp_advantage']>.01 for r in raw),
            same_length_wrong_gold_advantage_gt0p1=sum(r['same_length'] and not r['greedy_correct'] and r['sequence_logp_advantage']>.1 for r in raw),
            **{'same_length_wrong_'+k+'_correct':sum(r['same_length'] and not r['greedy_correct'] and r[k+'_correct'] for r in raw) for k in ('condition','first_digit','last_digit')},
            **{k+'_ties':sum(r[k+'_tie'] for r in raw) for k in ('condition','first_digit','last_digit')},
            groups=len(groups),**{k+'_target_joint':sum(all(g[i,j][k+'_correct'] for i in (0,1) for j in (0,1)) for g in groups.values()) for k in ('condition','first_digit','last_digit')}))
        for factor in ('question_index','context_index','level','greedy_correct'):
            for value in sorted({r[factor] for r in raw}):
                part=[r for r in raw if r[factor]==value]
                strata.append(dict(**meta,factor=factor,value=value,rows=len(part),condition_correct=sum(r['condition_correct'] for r in part),first_digit_correct=sum(r['first_digit_correct'] for r in part),last_digit_correct=sum(r['last_digit_correct'] for r in part),greedy_correct=sum(r['greedy_correct'] for r in part)))
        if prior_choice is not None:assert math.isclose(maximum_replay_difference,receipt['max_candidate_score_replay_difference'],rel_tol=0,abs_tol=1e-12)
        checks.append(dict(job=job['label'],split=split,scores_sha256=sha(p),rows=len(raw),max_gpu_candidate_score_replay_difference=receipt['max_candidate_score_replay_difference'],independent_cpu_candidate_replay_difference=maximum_replay_difference if prior_choice is not None else None))
assert len(checks)==10
for name,records in [('near_summary.csv',summaries),('near_strata.csv',strata)]:
    with (O/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(O/'near_cpu_audit.json').write_text(json.dumps(dict(status='verified',cells=10,test_accessed=False,files=checks),indent=2)+'\n',encoding='utf8')
print(json.dumps(summaries))
