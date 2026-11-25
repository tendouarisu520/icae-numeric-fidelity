"""Deterministic natural-language relation edits with complete reconstruction."""
import argparse
import collections
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

SOURCE_SHA='68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002'
PARTITIONS=('train','development','confirmation')
STOP=set('a an the in on at of to by for and or as is are was were be been being do does did which what who whom whose when where why how many much year years date name named called from with into during after before that this these those it its he she they his her their had has have will would can could should than then'.split())


def digest(value):return hashlib.sha256(value.encode('utf-8')).hexdigest()


def article_split(title):
    value=int(digest('v11-article-20261025\0'+title)[:8],16)%10
    return 'development' if value==0 else 'confirmation' if value==1 else 'train'


def words(text):return set(re.findall(r"[A-Za-z][A-Za-z'-]*",text.casefold()))-STOP


@lru_cache(maxsize=4096)
def answer_pattern(answer):return re.compile(r'(?<!\w)'+re.escape(answer)+r'(?!\w)',re.IGNORECASE)


def contains(text,answer):return bool(answer_pattern(answer).search(text))


def sentence_spans(text):
    start=0;result=[]
    for match in re.finditer(r'[.!?](?=\s|$)',text):
        end=match.end()
        prefix=text[:end]
        if match.group()=='.' and (re.search(r'\b(?:St|Mr|Mrs|Ms|Dr|Prof|Sr|Jr|No|Vol|vs|etc|Corp|Inc|Ltd|v|d|b|e\.g|i\.e)\.$',prefix,re.I)
                                  or re.search(r'\b(?:[A-Za-z]\.){2,}$',prefix)
                                  or re.search(r'\b[A-Z]\.$',prefix)):
            continue
        while start<end and text[start].isspace():start+=1
        if end>start:result.append((start,end))
        start=end
    while start<len(text) and text[start].isspace():start+=1
    if start<len(text):result.append((start,len(text)))
    return result


def question_kind(question):
    q=question.casefold()
    if re.search(r'\b(when|year|date)\b',q):return 'time'
    if re.match(r'\s*(who|whom|whose)\b',q):return 'person'
    if re.search(r'\bwhere\b|\b(?:what|which)\s+(?:city|country|place|location)\b',q):return 'location'
    if re.search(r'\bhow\s+(?:many|much)\b|\b(?:what|which)\s+number\b',q):return 'number'
    return None


def compatible(a,b,kind):
    if re.fullmatch(r'\d{4}',a):return bool(re.fullmatch(r'\d{4}',b))
    if kind in ('time','number') and bool(re.search(r'\d',a))!=bool(re.search(r'\d',b)):return False
    if kind in ('person','location') and a[0].isupper()!=b[0].isupper():return False
    return .5<=len(b.split())/max(1,len(a.split()))<=2


def numeric_replacement(original,candidate,kind):
    pattern=r'(?P<prefix>(?:[Tt]he\s+)?)(?P<number>\d[\d,]*(?:\.\d+)?)(?P<suffix>[^\d]*)'
    left=re.fullmatch(pattern,original);right=re.fullmatch(pattern,candidate)
    if not left or not right:return None
    a=float(left['number'].replace(',',''));b=float(right['number'].replace(',',''))
    if a==b or b<=0:return None
    suffix=left['suffix']
    if kind=='time':
        if not re.fullmatch(r'\d{4}',left['number']) or suffix not in ('','s'):return None
        if not b.is_integer() or not 1000<=b<=2099:return None
        if suffix=='s':
            if int(b)%10 or abs(b-a)>20:return None
        elif abs(b-a)>5:return None
    else:
        if abs(b-a)>max(.25 if '.' in left['number'] else 2,abs(a)*.25):return None
        if '%' in suffix and not 0<b<=100:return None
    if '.' in left['number']:
        value=format(b,'.'+str(len(left['number'].split('.')[-1]))+'f')
    else:
        if not b.is_integer():return None
        value=format(int(b),',') if ',' in left['number'] else str(int(b))
    result=left['prefix']+value+suffix
    return result if result!=original else None


def unique_answer(qa,context,spans):
    if qa.get('is_impossible') or not qa.get('answers'):return None
    values={answer.get('text') for answer in qa['answers']}
    if len(values)!=1:return None
    text=next(iter(values))
    if not isinstance(text,str) or not 2<=len(text)<=80 or text.strip()!=text or len(text.split())>12 or text.casefold()=='unknown':return None
    if text.count('(')!=text.count(')') or text.count('[')!=text.count(']'):return None
    if len(list(answer_pattern(text).finditer(context)))!=1:return None
    start=qa['answers'][0].get('answer_start')
    if type(start)is not int or context[start:start+len(text)]!=text:return None
    support=next(((a,b) for a,b in spans if a<=start and start+len(text)<=b),None)
    if support is None:return None
    sentence=context[support[0]:support[1]]
    if re.match(r'\s*(he|she|it|they|his|her|their|this|these|those|such)\b',sentence,re.I):return None
    return dict(text=text,start=start,end=start+len(text),support=support)


def cyclic(values,key,limit=256):
    if not values:return
    start=int(digest(key)[:12],16)%len(values)
    for offset in range(min(limit,len(values))):yield values[(start+offset)%len(values)]


def make_unit(title,paragraph,answer_pool,donor_pool,split):
    context=paragraph['context'];spans=sentence_spans(context)
    if split not in PARTITIONS or len(spans)<3:return None
    uid='v11-'+digest(title+'\0'+context)[:24]
    known=[];unknown=[]
    for qa in sorted(paragraph['qas'],key=lambda value:value['id']):
        if not isinstance(qa.get('question'),str) or not qa['question'].strip():continue
        if qa.get('is_impossible'):
            unknown.append(qa);continue
        if re.search(r'\b(?:one of|name one|some|an example|examples|how many thousands)\b',qa['question'],re.I):continue
        answer=unique_answer(qa,context,spans)
        if answer:known.append((qa,answer))
    if len(known)<2 or not unknown:return None
    for target,answer in known:
        kind=question_kind(target['question'])
        if kind not in ('time','number') or contains(target['question'],answer['text']):continue
        target_sentence=context[answer['support'][0]:answer['support'][1]]
        if len(re.findall(r'\b\d[\d,.]*(?:%|s)?\b',target_sentence))!=1:continue
        if kind=='time' and len(re.findall(r'\b\d{4}(?:s)?\b',target_sentence))!=1:continue
        if kind in ('time','number') and not re.search(r'\d|century|january|february|march|april|may|june|july|august|september|october|november|december',answer['text'],re.I):continue
        bank=answer_pool.get(kind,()) if isinstance(answer_pool,dict) else answer_pool
        for other,other_answer in known:
            if other['id']==target['id'] or other_answer['support']==answer['support']:continue
            if contains(other['question'],answer['text']) or contains(target['question'],other_answer['text']):continue
            irrelevant=[span for span in spans if span not in (answer['support'],other_answer['support']) and len(context[span[0]:span[1]].split())>=5]
            if not irrelevant:continue
            for impossible in unknown:
                query_words=words(' '.join(value['question'] for value in (target,other,impossible)))
                forbidden=[answer['text'],other_answer['text']]
                replacement=None
                for candidate in cyclic(bank,uid+'foil'):
                    if candidate.get('kind')!=kind or candidate.get('split')!=split or candidate.get('title')==title:continue
                    value=numeric_replacement(answer['text'],candidate['text'],kind)
                    if value is None:continue
                    if value.casefold() in {a.casefold() for a in forbidden} or value.casefold()=='unknown' or contains(context,value):continue
                    if any(contains(q['question'],value) for q in (target,other,impossible)):continue
                    replacement=dict(candidate,source_text=candidate['text'],text=value);break
                if replacement is None:continue
                forbidden.append(replacement['text'])
                def donor(span,label):
                    count=len(context[span[0]:span[1]].split())
                    for value in cyclic(donor_pool,uid+label):
                        text=value['text']
                        if value.get('split')!=split or value.get('title')==title or not .65<=len(text.split())/max(1,count)<=1.55:continue
                        if words(text)&query_words or any(contains(text,a) for a in forbidden):continue
                        return value
                    return None
                deleted_donor=donor(answer['support'],'delete')
                irrelevant_span=irrelevant[int(digest(uid+'irrelevant')[:8],16)%len(irrelevant)]
                control_donor=donor(irrelevant_span,'control')
                if deleted_donor is None or control_donor is None:continue
                replace=lambda span,value:context[:span[0]]+value+context[span[1]:]
                contexts=[context,replace((answer['start'],answer['end']),replacement['text']),
                          replace(answer['support'],deleted_donor['text']),replace(irrelevant_span,control_donor['text'])]
                unit=dict(id=uid,split=split,article=title,contexts=contexts,
                          queries=[value['question'] for value in (target,other,impossible)],
                          candidates=[answer['text'],replacement['text'],'UNKNOWN'],
                          target_answers=[answer['text'],replacement['text'],'UNKNOWN',answer['text']],
                          unaffected_answer=other_answer['text'],unknown_answer='UNKNOWN',
                          provenance=dict(recipe='v11-natural-relations-3',source_context_sha256=digest(context),
                                          question_ids=[value['id'] for value in (target,other,impossible)],
                                          answer_span=[answer['start'],answer['end']],target_support=list(answer['support']),
                                          unaffected_support=list(other_answer['support']),irrelevant_span=list(irrelevant_span),
                                          replacement=replacement,deleted_donor=deleted_donor,control_donor=control_donor))
                try:audit_unit(unit)
                except ValueError:continue
                return unit
    return None


def audit_unit(unit):
    if unit.get('split')not in PARTITIONS or len(unit['contexts'])!=4 or len(unit['queries'])!=3 or len(set(unit['provenance']['question_ids']))!=3:
        raise ValueError('Complete four-context/three-question unit required')
    a,b,unknown=unit['candidates'];context=unit['contexts'][0];p=unit['provenance']
    if unknown!='UNKNOWN' or unit['target_answers']!=[a,b,unknown,a] or unit['unknown_answer']!=unknown or len({a.casefold(),b.casefold(),unknown.casefold()})!=3:
        raise ValueError('Target labels differ from fixed edit semantics')
    if digest(context)!=p['source_context_sha256']:raise ValueError('Original context changed')
    edit=lambda span,text:context[:span[0]]+text+context[span[1]:]
    if unit['contexts'][1]!=edit(p['answer_span'],b) or unit['contexts'][2]!=edit(p['target_support'],p['deleted_donor']['text']) or unit['contexts'][3]!=edit(p['irrelevant_span'],p['control_donor']['text']):
        raise ValueError('Edited contexts do not reconstruct exactly')
    if context[p['answer_span'][0]:p['answer_span'][1]]!=a:raise ValueError('Original answer offset mismatch')
    if not contains(unit['contexts'][0],a) or not contains(unit['contexts'][1],b) or contains(unit['contexts'][1],a) or any(contains(unit['contexts'][2],v) for v in (a,b)) or not contains(unit['contexts'][3],a):
        raise ValueError('Target lexical edit integrity failed')
    support=context[p['unaffected_support'][0]:p['unaffected_support'][1]]
    if any(support not in value or not contains(value,unit['unaffected_answer']) for value in unit['contexts']):
        raise ValueError('Unaffected support was changed')
    if any(value['split']!=unit['split'] or value['title']==unit['article'] for value in (p['replacement'],p['deleted_donor'],p['control_donor'])):
        raise ValueError('Donor split/article boundary failed')
    return True


def build_candidates(input_path,output):
    input_path,output=Path(input_path),Path(output)
    roots=[Path(value) for value in ('D:/CodexStorage/research/memory-binding-20261021','E:/Research/ICAE-ConditionalMemory-20261021')]
    if input_path not in [root/'data/public-squad2-20261025/train-v2.0.json' for root in roots] or output not in [root/'data/v11-natural-candidates-20261025-r3' for root in roots] or output.exists():
        raise ValueError('Fixed public train source and new D/E candidates output required')
    content=input_path.read_bytes()
    if hashlib.sha256(content).hexdigest()!=SOURCE_SHA:raise ValueError('Official SQuAD2 train bytes differ')
    source=json.loads(content);answer_banks={s:collections.defaultdict(list) for s in PARTITIONS};donors={s:[] for s in PARTITIONS};paragraphs=[]
    for article in source['data']:
        title=article['title'];split=article_split(title)
        for index,paragraph in enumerate(article['paragraphs']):
            context=paragraph['context'];paragraphs.append((title,split,paragraph))
            for a,b in sentence_spans(context):
                text=context[a:b]
                if 5<=len(text.split())<=40 and text.count('(')==text.count(')') and text.count('[')==text.count(']'):
                    donors[split].append(dict(text=text,title=title,id=digest(title+'\0'+context)[:16]+':'+str(a),split=split))
            for qa in paragraph['qas']:
                if qa.get('is_impossible') or not qa.get('answers'):continue
                kind=question_kind(qa['question']);text=qa['answers'][0]['text']
                if kind in ('time','number') and 2<=len(text)<=80 and len(text.split())<=12:
                    answer_banks[split][kind].append(dict(text=text,kind=kind,title=title,id=qa['id'],split=split))
    units={split:[] for split in PARTITIONS};seen=set()
    for position,(title,split,paragraph) in enumerate(paragraphs,1):
        if position%1000==0:
            print(json.dumps(dict(phase='build_candidates',paragraphs_processed=position,total=len(paragraphs),built=sum(map(len,units.values())))),flush=True)
        if len(paragraph['context'].split())>350:continue
        value=make_unit(title,paragraph,answer_banks[split],donors[split],split)
        if value and value['id']not in seen:
            seen.add(value['id']);units[split].append(value)
    output.mkdir(parents=True,exist_ok=False);files={}
    for split in PARTITIONS:
        path=output/(split+'.jsonl');values=sorted(units[split],key=lambda unit:digest(unit['id']+'v11-order'))
        with path.open('x',encoding='utf-8') as handle:
            for value in values:handle.write(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n')
        files[split]=dict(units=len(values),articles=len({v['article'] for v in values}),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result=dict(status='candidate_data_built_semantic_and_token_audit_pending',source_sha256=SOURCE_SHA,
                partitions=files,source_paragraphs=len(paragraphs),recipe='v11-natural-relations-3',
                target_kinds=['time','number'],numeric_units_preserved=True,
                donor_partition_isolated=True,contexts_per_unit=4,questions_per_context=3,
                official_external_dev_used=False,internal_confirmation_or_reserved_test_accessed=False)
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(build_candidates(args.input,args.output)),flush=True)


if __name__=='__main__':main()
