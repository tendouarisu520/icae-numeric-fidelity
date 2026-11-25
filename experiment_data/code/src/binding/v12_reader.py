"""Context-only RAM memories, matched readers, and refusal diagnostics."""
import hashlib
import json
import time
from pathlib import Path
from .v11_pipeline import prediction_rows,read_rows,write_json,sha,normalize,em_f1
from .v11_error_analysis import describe,indexed
from .v12_protocol import banned_next,calibrate


class RamMemories:
    def __init__(self,entries,token_ids,ft):
        self.entries=entries;self.token_ids=token_ids;self.ft=ft

    def get(self,ids):
        from .v4_model import context_key
        key=context_key(ids)
        if key not in self.entries or self.token_ids[key]!=ids:raise ValueError('Exact context absent from frozen cache')
        return self.entries[key]

    def digest(self):
        from .repro_probe import _tensor_bytes
        h=hashlib.sha256()
        for key,memory in sorted(self.entries.items()):h.update(key.encode());h.update(_tensor_bytes(memory))
        h.update(_tensor_bytes(self.ft));return h.hexdigest()


def build_ram_cache(encoder,contexts,progress=None):
    import torch
    from .v4_model import context_key
    if any(p.requires_grad for p in encoder.parameters()) or any(m.training for m in encoder.modules()):
        raise ValueError('Completely frozen eval encoder required')
    unique={context_key(ids):list(ids) for ids in contexts};entries={}
    if not unique:raise ValueError('Empty contexts')
    with torch.no_grad():
        for i,(key,ids) in enumerate(sorted(unique.items()),1):
            memory=encoder.compress_ids(torch.tensor([ids],device=encoder.device)).detach().cpu().contiguous()
            if memory.shape!=(1,128,encoder.hidden_size) or not bool(torch.isfinite(memory).all()):
                raise ValueError('Invalid128-slot frozen memory')
            entries[key]=memory
            if progress is not None and (i==1 or i%64==0 or i==len(unique)):progress(cache_completed=i,cache_total=len(unique))
        ft=encoder.memory_token_embed.weight[130].detach().cpu().clone().contiguous()
    return RamMemories(entries,unique,ft)


def greedy(model,inputs,tokenizer,banned=()):
    import torch
    generated=[];cache=None
    with torch.no_grad():
        for _ in range(32):
            out=model.forward_prefix(inputs,past_key_values=cache,use_cache=True,return_dict=True)
            logits=out.logits[0,-1].float().clone()
            for token in banned_next(generated,banned):logits[token]=-torch.inf
            token=int(logits.argmax());generated.append(token)
            if token==tokenizer.eos_token_id:break
            cache=out.past_key_values;inputs=model.embeddings([token]).unsqueeze(0)
    return tokenizer.decode(generated,skip_special_tokens=True),dict(token_ids=generated,eos_emitted=generated[-1]==tokenizer.eos_token_id)


def score_ids(model,prefix,answer):
    import torch
    if not answer:raise ValueError('Target+EOS required')
    with torch.no_grad():
        inputs=torch.cat((prefix,model.embeddings(answer).unsqueeze(0)),dim=1)
        logits=model.forward_prefix(inputs,use_cache=False,return_dict=True).logits[0,prefix.shape[1]-1:prefix.shape[1]-1+len(answer)].float()
        values=logits.log_softmax(-1).gather(-1,torch.tensor(answer,device=model.device).unsqueeze(-1)).squeeze(-1)
        if not bool(torch.isfinite(values).all()):raise FloatingPointError('Nonfinite answer scores')
        return dict(mean=float(values.mean()),sum=float(values.sum()),tokens=len(answer),first=float(values[0]),eos=float(values[-1]))


def prefix_for(model,item,i,j,cache,mode='matched',donor=None):
    import torch
    query=item['queries'][j];ids=item['contexts'][i]
    if mode=='full':return model.prefix(dict(context=ids,**query),'full_reader')
    memory=cache.get((donor or item)['contexts'][i]).to(model.device)
    if mode=='null':memory=torch.zeros_like(memory)
    if mode not in ('matched','shuffled','null'):raise ValueError('Unknown memory intervention')
    return model.prefix(dict(context=ids,**query),'memory_reader',memory)


def evaluate(model,tokenizer,units,prepared,cache,output,mode='matched',donors=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);rows=[];started=time.perf_counter()
    with (output/'predictions.jsonl').open('x',encoding='utf8',buffering=1) as handle:
        for unit in units:
            item=prepared[unit['id']];values=[];details=[]
            donor=prepared[donors[unit['id']]] if mode=='shuffled' else None
            for i in range(4):
                predictions=[]
                for j in range(3):
                    p,detail=greedy(model,prefix_for(model,item,i,j,cache,mode,donor),tokenizer)
                    predictions.append(p);details.append(detail)
                values.append(predictions)
            group=prediction_rows(unit,values)
            for row,detail in zip(group,details):
                row['generation']=detail;handle.write(json.dumps(row,ensure_ascii=False)+'\n')
            rows.extend(group)
    result=describe(rows);result.update(wall_seconds=time.perf_counter()-started,predictions_sha256=sha(output/'predictions.jsonl'),memory_mode=mode)
    write_json(output/'metrics.json',result)
    if donors is not None:write_json(output/'donors.json',donors)
    return result,rows


def require_reproduction(actual,expected):
    a,b=indexed(actual),indexed(expected)
    if set(a)!=set(b):raise ValueError('Source replay coverage mismatch')
    mismatches=[key for key in a if a[key]['gold']!=b[key]['gold'] or a[key]['prediction']!=b[key]['prediction'] or a[key]['generation']!=b[key]['generation']]
    if mismatches:raise RuntimeError('Exact source generation mismatch:'+repr(mismatches[:4]))
    return dict(exact=True,rows=len(a))


def reader_backward(model,item,cache):
    value=0.;gold=(0,1,2,0)
    for i in range(4):
        for j in range(3):
            answer=item['candidates'][gold[i] if j==0 else 2] if j!=1 else item['unaffected_answer']
            loss=model.answer_nll(prefix_for(model,item,i,j,cache),answer)/12
            value+=float(loss.detach());loss.backward()
    return dict(loss=value,sft_loss=value,supervised_answers=12)


def diagnose(model,tokenizer,units,prepared,cache,raw,output,mode='matched',common=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);index=indexed(raw);records=[]
    banned=[tokenizer.encode(text,add_special_tokens=False) for text in (' UNKNOWN','UNKNOWN',' unknown','unknown',' Unknown','Unknown')]
    encode=lambda text:tokenizer.encode(' '+text.strip(),add_special_tokens=False)+[tokenizer.eos_token_id]
    gold_indices=(0,1,2,0);started=time.perf_counter()
    with (output/'scores.jsonl').open('x',encoding='utf8',buffering=1) as handle:
        for unit in units:
            item=prepared[unit['id']]
            for i in range(4):
                for j in range(3):
                    key=(unit['id'],i,j);row=index[key];prefix=prefix_for(model,item,i,j,cache,mode)
                    if mode=='full' and common is not None:alternative=common[key]
                    elif normalize(row['prediction'])!='unknown':alternative=row['prediction']
                    else:alternative=greedy(model,prefix,tokenizer,banned)[0]
                    valid=bool(normalize(alternative)) and normalize(alternative)!='unknown'
                    gold=item['unaffected_answer'] if j==1 else item['candidates'][gold_indices[i] if j==0 else 2]
                    memo={}
                    def score(ids):
                        k=tuple(ids)
                        if k not in memo:memo[k]=score_ids(model,prefix,ids)
                        return memo[k]
                    u=score(item['candidates'][2]);g=score(gold);a=score(encode(alternative)) if valid else None
                    value=dict(unit_id=unit['id'],context_index=i,question_index=j,gold=row['gold'],prediction=row['prediction'],
                        alternative=alternative,alternative_valid=valid,unknown_score=u,gold_score=g,alternative_score=a,
                        margin=a['mean']-u['mean'] if a else None,sum_margin=a['sum']-u['sum'] if a else None,
                        oracle_gold_margin=g['mean']-u['mean'] if row['gold']!='UNKNOWN' else None)
                    if common is not None:
                        value['common_candidate']=common[key];value['common_score']=score(encode(common[key]))
                    handle.write(json.dumps(value,allow_nan=False)+'\n');records.append(value)
    articles={unit['id']:unit['article'] for unit in units}
    calibration=calibrate(records,articles) if len(set(articles.values()))>=2 else None
    if calibration:
        from .v11_pipeline import metrics
        calibrated=[dict(row,prediction=p) for row,p in zip(raw,calibration['out_of_fold_predictions'])]
        calibration['out_of_fold_metrics']=metrics(calibrated)
        write_json(output/'calibration.json',calibration)
    known_refused=[r for r in records if r['gold']!='UNKNOWN' and normalize(r['prediction'])=='unknown']
    summary=dict(rows=len(records),wall_seconds=time.perf_counter()-started,scoring='Answer+EOS; mean and sum log probability',
        forced_known_recovered=[sum(em_f1(r['alternative'],r['gold'])[0] for r in known_refused),len(known_refused)],
        greedy_unknown_with_positive_gold_margin=[sum(r['oracle_gold_margin']>0 for r in known_refused),len(known_refused)],
        scores_sha256=sha(output/'scores.jsonl'),calibration=calibration,
        oracle_scores_are_diagnostic_only=True,raw_greedy_primary=True,confirmation_evaluated=False,reserved_test_accessed=False)
    write_json(output/'summary.json',summary)
    return summary,{(r['unit_id'],r['context_index'],r['question_index']):r['alternative'] for r in records}
