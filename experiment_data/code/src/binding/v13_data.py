"""Answerable natural QA and explicitly synthetic multi-condition probes."""
from collections import OrderedDict
from functools import lru_cache
import hashlib
import random

TEAMS={'train':('Birch','Cedar','Maple','Oak','Pine','Willow','Ash','Alder'),
       'development':('Atlas','Orion','Vega','Lyra','Nova','Draco','Aquila','Cygnus')}
FIELDS=('badge','locker','route')
MODES=(('day','night'),('dry','wet'),('warm','cold'),('fast','slow'))

def digest(text):return hashlib.sha256(text.encode()).hexdigest()

@lru_cache(maxsize=2)
def value_pool(split):
    if split not in TEAMS:raise ValueError('Train/development only')
    return tuple(str(x) for x in range(1000,10000) if (int(digest('v13-label:'+str(x))[:8],16)%5==0)==(split=='development'))

def render(facts,background):
    groups=OrderedDict()
    for fact in facts:
        team,mode,field=fact['key'];groups.setdefault((team,mode),{})[field]=fact['value']
    lines=['Record entries apply only to their named team and operating mode.']
    if background:lines.append('Background: '+background)
    for (team,mode),fields in groups.items():
        lines.append('Team '+team+', '+mode+' mode: '+', '.join(f+' '+fields[f] for f in FIELDS)+'.')
    return '\n'.join(lines)

def make_probe(source,split,level):
    if split not in TEAMS or level not in (2,3,4):raise ValueError('Fixed train/development and2/3/4teams required')
    rng=random.Random(int(digest('v13-scene:'+source['id'])[:16],16))
    entities=rng.sample(list(TEAMS[split]),4)[:level];modes=MODES[rng.randrange(len(MODES))]
    labels=rng.sample(value_pool(split),25)
    facts=[]
    for team in entities:
        for mode in modes:
            for field in FIELDS:facts.append(dict(key=[team,mode,field],value=labels[len(facts)]))
    qkeys=[[entities[0],modes[0],'badge'],[entities[0],modes[1],'badge'],
           [entities[0],modes[0],'locker'],[entities[1],modes[0],'badge']]
    swapped=[dict(key=list(f['key']),value=f['value']) for f in facts]
    indices=[next(i for i,f in enumerate(facts) if f['key']==key) for key in qkeys[:2]]
    swapped[indices[0]]['value'],swapped[indices[1]]['value']=swapped[indices[1]]['value'],swapped[indices[0]]['value']
    unrelated=[dict(key=list(f['key']),value=f['value']) for f in facts]
    edit=next(f for f in unrelated if f['key']==[entities[1],modes[1],'route']);edit['value']=labels[-1]
    variants=[facts,swapped,unrelated,list(reversed(facts))]
    # A short natural paragraph is background, not evidence for the generated records.
    background=' '.join(source['contexts'][0].split())[:180]
    queries=[f'For team {team} in {mode} mode, what is the {field} code?' for team,mode,field in qkeys]
    answers=[]
    for variant in variants:
        mapping={tuple(f['key']):f['value'] for f in variant}
        answers.append([mapping[tuple(k)] for k in qkeys])
    return dict(id=source['id']+'-probe-'+str(level),source_id=source['id'],article=source['article'],family='probe',level=level,
        entities=entities,modes=modes,facts=variants,query_keys=qkeys,contexts=[render(v,background) for v in variants],
        queries=queries,answers=answers,split=split,synthetic_relation_records=True)

def make_natural(source):
    answers=[[source['target_answers'][i] if i!=2 else None,source['unaffected_answer']] for i in range(4)]
    if any(v=='UNKNOWN' for row in answers for v in row if v is not None):raise ValueError('Answerable labels only')
    return dict(id=source['id']+'-natural',source_id=source['id'],article=source['article'],family='natural',level=0,
        contexts=list(source['contexts']),queries=list(source['queries'][:2]),answers=answers,synthetic_relation_records=False)

def prepare(unit,tokenizer):
    from .v11_pipeline import PREFIX,SUFFIX
    encode=lambda s:tokenizer.encode(s,add_special_tokens=False)
    contexts=[[tokenizer.bos_token_id]+encode(c) for c in unit['contexts']]
    if any(not 1<=len(c)<=512 for c in contexts):raise ValueError('Context exceeds512; relation records must not be truncated')
    instruction='Answer using only the passage. Give only the short answer.\nQuestion: '
    queries=[dict(left=list(PREFIX),right=encode(instruction+q)+list(SUFFIX)) for q in unit['queries']]
    if any(len(q['right'])>128 for q in queries):raise ValueError('Query exceeds128')
    answers=[[encode(' '+v)+[tokenizer.eos_token_id] if v is not None else None for v in row] for row in unit['answers']]
    if any(not 2<=len(a)<=25 for row in answers for a in row if a is not None):raise ValueError('Answer token budget')
    if unit['family']=='probe' and len({len(a) for row in answers for a in row})!=1:
        raise ValueError('Controlled code candidates must have equal token lengths')
    return dict(id=unit['id'],family=unit['family'],contexts=contexts,queries=queries,answers=answers)
