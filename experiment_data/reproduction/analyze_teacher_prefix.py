"""Gold-prefix oracle diagnostic from already recorded candidate token scores."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json
R=Path(__file__).resolve().parents[1];O=R/'followup';P=R
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out=[];checks=[]
for rp in sorted((O/'choice').glob('*/*/receipt.json')):
    rec=read(rp);split,name=rec['split'],rec['job'];p=rp.parent/'scores.jsonl';assert sha(p)==rec['scores_sha256']
    oldpath=P/'confirmation'/name/'predictions.jsonl' if split=='confirmation' else next((P/'evidence/recorded/runs').glob(f'*/{name}/evaluation-512/predictions.jsonl'))
    assert sha(oldpath)==rec['raw_predictions_sha256'];old={(r['unit_id'],r['context_index'],r['question_index']):r for r in rows(oldpath)}
    data=rows(p);_,arm,method,s=name.split('-');seed=int(s[4:]);same_prefix=0;different_argmax=0;digit_map={};aligned=[]
    for r in data:
        ids=r['candidate_scores'][r['question_index']]['token_ids'];pred=r['candidate_scores'][r['question_index']]['teacher_forced_argmax']
        actual=old[r['unit_id'],r['context_index'],r['question_index']]['generation']['token_ids']
        assert len(ids)==len(pred)==6 and ids[0]==28705 and ids[-1]==2
        for char,token in zip(r['gold'],ids[1:5]):
            if char in digit_map:assert digit_map[char]==token
            digit_map[char]=token
        r['same_prefix_discrepancy']=False
        for j in range(min(len(ids),len(actual))):
            if actual[:j]==ids[:j]:
                same_prefix+=1;different_argmax+=int(pred[j]!=actual[j])
                r['same_prefix_discrepancy'] |= pred[j]!=actual[j]
        aligned.append((r,ids,pred,actual))
    for subset in ('all','wrong_generation','wrong_generation_consistent_prefix'):
        selected=[x for x in aligned if subset=='all' or (not x[0]['greedy_correct'] and (subset=='wrong_generation' or not x[0]['same_prefix_discrepancy']))]
        if not selected:continue
        for pos in range(4):
            n=len(selected);digit=pos+1
            greedy_token=sum(len(actual)>digit and actual[digit]==ids[digit] for _,ids,pred,actual in selected)
            forced=sum(pred[digit]==ids[digit] for _,ids,pred,actual in selected)
            damaged_prefix=[x for x in selected if x[3][:digit]!=x[1][:digit]]
            out.append(dict(split=split,arm=arm,method=method,seed=seed,subset=subset,digit=pos+1,n=n,greedy_token_correct=greedy_token,gold_prefix_argmax_correct=forced,
                earlier_greedy_prefix_wrong=len(damaged_prefix),gold_prefix_correct_after_earlier_error=sum(x[2][digit]==x[1][digit] for x in damaged_prefix),
                oracle_correct_greedy_wrong=sum(pred[digit]==ids[digit] and (len(actual)<=digit or actual[digit]!=ids[digit]) for _,ids,pred,actual in selected),
                greedy_correct_oracle_wrong=sum(pred[digit]!=ids[digit] and len(actual)>digit and actual[digit]==ids[digit] for _,ids,pred,actual in selected)))
    checks.append(dict(split=split,job=name,same_prefix_token_comparisons=same_prefix,different_argmax_with_identical_prefix=different_argmax,rows_with_same_prefix_discrepancy=sum(x[0]['same_prefix_discrepancy'] for x in aligned),gold_digit_token_map=digit_map))
with (O/'teacher_prefix_summary.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
(O/'teacher_prefix_audit.json').write_text(json.dumps(dict(status='descriptive_oracle_diagnostic',cells=20,files=checks,note='Teacher-forced gold prefixes are not available to an autonomous decoder; same-prefix discrepancies audit cached versus uncached inference.'),indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(same_prefix=sum(r['same_prefix_token_comparisons'] for r in checks),discrepancies=sum(r['different_argmax_with_identical_prefix'] for r in checks),sft=[r for r in out if r['arm']=='compressed' and r['method']=='sft' and r['subset']=='wrong_generation'])))
