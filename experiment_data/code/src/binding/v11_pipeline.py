"""Natural QA token admission, article-balanced sampling, and edit metrics."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import string

PREFIX=[1,733,16289,28793]
SUFFIX=[733,28748,16289,28793]
INSTRUCTION=('Answer the question using only the passage. If the passage does not contain '
             'the answer, output UNKNOWN. Give only the short answer.\nQuestion: ')
MANUAL_EXCLUSIONS={
    'v11-419693c612d8dea3cd12a106':'Original impossible question paraphrases political anthropology; annotation ambiguous.',
    'v11-1932b9ae2e2fdf4c93a888ab':'Unaffected question asks when created; annotated1926 is conception versus1947 implementation.',
    'v11-8f3be18e7da7d4429867c81d':'Question asks dam water; annotated80percent refers to groundwater.',
    'v11-995bc2364cbb3716e9e8677b':'Question associates1920 land with8million; source says1.5million.',
    'v11-a4d73f6d099109258f713702':'Question conflates two cornerstone publications with15 theses.',
    'v11-a3dec41fc9873d632a5347b9':'Official impossible RAF founding year can be inferred from dated appointment plus two months.'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');temp.replace(path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def prepare_unit(unit,tokenizer):
    encode=lambda value:tokenizer.encode(value,add_special_tokens=False)
    contexts=[[tokenizer.bos_token_id]+encode(text) for text in unit['contexts']]
    queries=[dict(left=list(PREFIX),right=encode(INSTRUCTION+query)+SUFFIX) for query in unit['queries']]
    candidates=[encode(' '+value)+[tokenizer.eos_token_id] for value in unit['candidates']]
    other=encode(' '+unit['unaffected_answer'])+[tokenizer.eos_token_id]
    if len(contexts)!=4 or len(queries)!=3 or len(candidates)!=3:
        raise ValueError('Incomplete relation unit')
    if any(not 1<=len(value)<=512 for value in contexts) or any(len(value['right'])>128 for value in queries):
        raise ValueError('Input token budget exceeded; truncation forbidden')
    if any(not 2<=len(value)<=25 for value in candidates+[other]) or len({tuple(x) for x in candidates})!=3:
        raise ValueError('Short distinct answer+EOS tokens required')
    return dict(id=unit['id'],contexts=contexts,queries=queries,candidates=candidates,unaffected_answer=other)


def balanced_select(rows,count,per_article):
    groups=collections.OrderedDict()
    for row in rows:groups.setdefault(row['article'],[]).append(row)
    result=[]
    for level in range(per_article):
        for values in groups.values():
            if level<len(values):result.append(values[level])
            if len(result)==count:return result
    raise ValueError(f'Only{len(result)} units under article cap; need{count}')


def admit(root):
    from transformers import AutoTokenizer
    from .v11_data import audit_unit
    root=Path(root)
    if str(root).replace('\\','/').lower()!='e:/research/icae-conditionalmemory-20261021':
        raise ValueError('Remote E root required')
    source=root/'data/v11-natural-candidates-20261025-r3';output=root/'data/v11-natural-admitted-20261025-r3'
    output.mkdir(exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(str(root/'models/mistral'),local_files_only=True,use_fast=False)
    result=dict(status='admitted_for_semantic_review',source_manifest_sha256=sha(source/'manifest.json'),partitions={},manual_exclusions=MANUAL_EXCLUSIONS)
    for split in ('train','development','confirmation'):
        accepted=[];rejected=collections.Counter()
        for unit in read_rows(source/(split+'.jsonl')):
            audit_unit(unit)
            if unit['id'] in MANUAL_EXCLUSIONS:rejected['manual_annotation_ambiguity']+=1;continue
            try:prepared=prepare_unit(unit,tokenizer)
            except ValueError as error:rejected[str(error)]+=1;continue
            unit['token_lengths']=dict(contexts=list(map(len,prepared['contexts'])),
                                       questions=[len(q['right']) for q in prepared['queries']],
                                       answers=list(map(len,prepared['candidates']+[prepared['unaffected_answer']])))
            accepted.append(unit)
        path=output/(split+'.jsonl')
        path.write_text(''.join(json.dumps(u,ensure_ascii=False,separators=(',',':'))+'\n' for u in accepted),encoding='utf8')
        result['partitions'][split]=dict(units=len(accepted),articles=len({u['article'] for u in accepted}),
                                         rejected=dict(rejected),sha256=sha(path))
    write_json(output/'manifest.json',result);return result


def normalize(value):
    value=value.casefold().translate(str.maketrans('','',string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',value).split())


def em_f1(prediction,gold):
    prediction,gold=normalize(prediction),normalize(gold)
    if gold=='unknown':
        value=int(prediction==gold);return value,float(value)
    exact=int(prediction==gold)
    left,right=prediction.split(),gold.split()
    common=sum((collections.Counter(left)&collections.Counter(right)).values())
    return exact,(2.*common/(len(left)+len(right)) if common else 0.)


def prediction_rows(unit,predictions):
    if len(predictions)!=4 or any(len(value)!=3 for value in predictions):raise ValueError('Twelve outputs required')
    return [dict(unit_id=unit['id'],context_index=i,question_index=j,
                 prediction=predictions[i][j],gold=(unit['target_answers'][i] if j==0 else unit['unaffected_answer'] if j==1 else 'UNKNOWN'),
                 candidates=unit['target_answers'][:2]) for i in range(4) for j in range(3)]


def metrics(rows):
    units=collections.defaultdict(list)
    if not rows:raise ValueError('Nonempty predictions required')
    for row in rows:units[row['unit_id']].append(row)
    counts={};scores=[em_f1(row['prediction'],row['gold']) for row in rows]
    counts['exact']=[sum(x[0] for x in scores),len(rows)]
    for name,predicate in (
        ('known',lambda r:r['gold']!='UNKNOWN'),('unknown',lambda r:r['gold']=='UNKNOWN'),
        ('genuine_unknown',lambda r:r['question_index']==2),
        ('deleted_target_unknown',lambda r:r['context_index']==2 and r['question_index']==0)):
        subset=[row for row in rows if predicate(row)]
        counts[name+'_exact']=[sum(em_f1(r['prediction'],r['gold'])[0] for r in subset),len(subset)]
    known=[r for r in rows if r['gold']!='UNKNOWN']
    targets=[r for r in known if r['question_index']==0]
    counts['known_false_unknown']=[sum(normalize(r['prediction'])=='unknown' for r in known),len(known)]
    counts['known_target_misbinding']=[sum(any(normalize(r['prediction'])==normalize(c) and normalize(c)!=normalize(r['gold']) for c in r['candidates']) for r in targets),len(targets)]
    counts.update({name:[0,len(units)] for name in ('target_all_four','unaffected_all_four','unaffected_consistent')})
    for values in units.values():
        if len(values)!=12 or {(r['context_index'],r['question_index']) for r in values}!={(i,j) for i in range(4) for j in range(3)}:
            raise ValueError('Complete unique twelve-row groups required')
        target=[r for r in values if r['question_index']==0];other=[r for r in values if r['question_index']==1]
        counts['target_all_four'][0]+=int(all(em_f1(r['prediction'],r['gold'])[0] for r in target))
        counts['unaffected_all_four'][0]+=int(all(em_f1(r['prediction'],r['gold'])[0] for r in other))
        counts['unaffected_consistent'][0]+=int(len({normalize(r['prediction']) for r in other})==1)
    return dict(counts=counts,rates={name:a/b for name,(a,b) in counts.items()},
                f1=sum(x[1] for x in scores)/len(rows),units=len(units),rows=len(rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    print(json.dumps(admit(parser.parse_args().root)),flush=True)
