"""Frozen-official-memory, answerable mixed QA, reader-only controlled study."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
from .v11_pipeline import read_rows,sha,write_json
from .v13_data import make_natural,make_probe,prepare,digest
from .v13_protocol import metrics,prediction_rows,learning_rate

ROOT=Path('E:/Research/ICAE-ConditionalMemory-20261021')
DATA=ROOT/'data/v13-answerable-relations-20261026'
RUNS=ROOT/'runs/v13-answerable-reading-20261026'
SOURCE=ROOT/'data/v11-natural-frozen-20261025'

def build_units(rows,split):
    if split not in ('train','development'):raise ValueError('Train/development only')
    result=[]
    for u in rows:
        result.append(make_natural(u))
        levels=(2+int(digest(u['id'])[:8],16)%3,) if split=='train' else (2,3,4)
        result.extend(make_probe(u,split,level) for level in levels)
    return result

def prepare_data():
    from transformers import AutoTokenizer
    if DATA.exists():raise ValueError('Preserve existing data; do not overwrite')
    source=json.loads((SOURCE/'manifest.json').read_text(encoding='utf8'))
    rows={s:read_rows(SOURCE/(s+'.jsonl')) for s in ('train','development')}
    if (len(rows['train']),len(rows['development']))!=(512,32):raise ValueError('Fixed512/32 source required')
    if {u['article'] for u in rows['train']}&{u['article'] for u in rows['development']}:raise ValueError('Article leakage')
    for s in rows:
        if sha(SOURCE/(s+'.jsonl'))!=source['partitions'][s]['sha256']:raise ValueError('Original source changed')
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/mistral'),local_files_only=True,use_fast=False)
    DATA.mkdir();manifest=dict(schema=13,source_manifest_sha256=sha(SOURCE/'manifest.json'),
        source_hashes={s:sha(SOURCE/(s+'.jsonl')) for s in rows},partitions={},overfit_ids=source['overfit_ids'],
        encoder_input='Context tokens only',controlled_records_are_synthetic=True,
        confirmation_loaded=False,reserved_test_accessed=False)
    for split,values in rows.items():
        units=build_units(values,split);lengths=[];answers=0
        with (DATA/(split+'.jsonl')).open('x',encoding='utf8') as f:
            for unit in units:
                item=prepare(unit,tokenizer);lengths.extend(map(len,item['contexts']))
                answers+=sum(a is not None for row in item['answers'] for a in row)
                f.write(json.dumps(unit,ensure_ascii=False)+'\n')
        manifest['partitions'][split]=dict(sha256=sha(DATA/(split+'.jsonl')),sources=len(values),units=len(units),
            natural_units=sum(u['family']=='natural' for u in units),probe_units=sum(u['family']=='probe' for u in units),
            articles=len({u['article'] for u in values}),known_labels=answers,context_tokens_min=min(lengths),
            context_tokens_max=max(lengths),context_tokens_mean=sum(lengths)/len(lengths))
    write_json(DATA/'manifest.json',manifest);print(json.dumps(manifest),flush=True)

def evaluate(model,tokenizer,units,items,cache,arm,output,progress=None):
    from .v12_reader import greedy
    from .v13_objectives import prefix
    output=Path(output);output.mkdir(parents=True,exist_ok=False);rows=[];started=time.perf_counter()
    with (output/'predictions.jsonl').open('x',encoding='utf8',buffering=1) as f:
        for number,u in enumerate(units,1):
            item=items[u['id']];predictions=[];generation={}
            for i,row in enumerate(item['answers']):
                values=[]
                for j,a in enumerate(row):
                    if a is None:values.append(None);continue
                    p,g=greedy(model,prefix(model,item,i,j,cache,arm),tokenizer)
                    values.append(p);generation[i,j]=g
                predictions.append(values)
            group=prediction_rows(u,predictions)
            for row in group:
                row['generation']=generation[row['context_index'],row['question_index']]
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
            rows.extend(group)
            if progress is not None and (number==1 or number%16==0 or number==len(units)):
                progress(evaluation_units=number,evaluation_total=len(units))
    result={family:metrics([r for r in rows if r['family']==family]) for family in ('natural','probe')}
    result['probe_by_level']={str(k):metrics([r for r in rows if r['family']=='probe' and r['level']==k])
        for k in (2,3,4) if any(r['family']=='probe' and r['level']==k for r in rows)}
    result.update(wall_seconds=time.perf_counter()-started,predictions_sha256=sha(output/'predictions.jsonl'),
        generation_without_eos=[sum(not r['generation']['eos_emitted'] for r in rows),len(rows)])
    write_json(output/'metrics.json',result);return result

def selection_key(result):
    probe=sum(v['rates']['target_joint'] for v in result['probe_by_level'].values())/len(result['probe_by_level'])
    return .5*probe+.5*result['natural']['known_f1']

def gate(result):
    return (result['natural']['rates']['known_exact']>=.90 and result['probe']['rates']['known_exact']>=.95
        and result['probe']['rates']['target_joint']>=.875
        and result['probe']['rates']['locality_joint']>=.875)

def train(args):
    from .repro_probe import configure_environment,configure_torch,runtime_settings
    configure_environment('det_auto');configure_torch('det_auto')
    import numpy as np
    import torch
    from safetensors.torch import save_file,load_file
    from .model import load_icae
    from .v11_training import load_model,schedule,generate
    from .v12_reader import build_ram_cache,greedy
    from .v13_objectives import backward,prefix
    from .v9_model import adapter_state,load_state
    from .v4_model import state_identity
    if ROOT.drive.lower()!='e:':raise ValueError('Remote E only')
    if args.arm=='full' and args.method!='sft':raise ValueError('Full SFT reference only')
    if args.phase=='smoke' and (args.arm,args.method,args.seed)!=('compressed','relation',73):raise ValueError('Fixed smoke')
    if args.phase=='overfit' and (args.method,args.seed)!=('sft',73):raise ValueError('Fixed overfit')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    name=f'{args.phase}-{args.arm}-{args.method}-seed{args.seed}'
    output=RUNS/name;output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    state=dict(status='running',phase='preflight',job=name,pid=os.getpid(),optimizer_steps=0)
    def update(**values):
        state.update(values,updated=time.time());write_json(output/'status.json',state);print(json.dumps(state),flush=True)
    update()
    try:
        manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf8'))
        splits=('train','development') if args.phase=='main' else ('train',)
        for split in splits:
            if sha(DATA/(split+'.jsonl'))!=manifest['partitions'][split]['sha256']:raise ValueError('Frozen V13 data changed')
        train_units=read_rows(DATA/'train.jsonl');by_source=defaultdict(list)
        for u in train_units:by_source[u['source_id']].append(u)
        if len(by_source)!=512 or any(len(v)!=2 for v in by_source.values()):raise ValueError('Fixed512 paired sources')
        overfit_ids=manifest['overfit_ids'][:2 if args.phase=='smoke' else 8]
        if args.phase!='main':
            by_source={k:by_source[k] for k in overfit_ids};train_units=[u for units in by_source.values() for u in units]
        development=read_rows(DATA/'development.jsonl') if args.phase=='main' else train_units
        max_steps={'smoke':2,'overfit':1024,'main':512}[args.phase]
        order=schedule(list(by_source),args.seed,max_steps)
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/mistral'),local_files_only=True,use_fast=False)
        items={u['id']:prepare(u,tokenizer) for u in train_units+development}
        frozen_cache=None;cache_digest=None;replay=[];writer_peak=None
        source_paths=[Path(__file__),ROOT/'src/binding/v13_data.py',ROOT/'src/binding/v13_protocol.py',
            ROOT/'src/binding/v13_objectives.py',ROOT/'src/binding/v12_reader.py',ROOT/'src/binding/v4_model.py',DATA/'manifest.json']
        source_paths.extend(DATA/(s+'.jsonl') for s in splits)
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
        if args.arm=='compressed':
            update(phase='load_official_encoder')
            encoder,enc_tokenizer=load_icae(str(ROOT/'models/mistral'),str(ROOT/'models/icae/mistral_7b_ft_icae.safetensors'),
                device='cuda',dtype=torch.bfloat16)
            encoder.requires_grad_(False);encoder.eval();versions={n:p._version for n,p in encoder.named_parameters()}
            update(phase='encode_context_only_memories');torch.cuda.reset_peak_memory_stats()
            frozen_cache=build_ram_cache(encoder,[c for item in items.values() for c in item['contexts']],update)
            cache_digest=frozen_cache.digest();writer_peak=torch.cuda.max_memory_allocated()/1024**2
            if versions!={n:p._version for n,p in encoder.named_parameters()}:raise RuntimeError('Encoder changed')
            # Actual7B zero-reader replay on one natural and one controlled prefix.
            for unit in train_units[:2]:
                item=items[unit['id']];memory=frozen_cache.get(item['contexts'][0]).to(encoder.device)
                q=item['queries'][0];p=encoder.prefix_ids(memory,q['left'],q['right'])
                replay.append((unit['id'],generate(encoder,p,enc_tokenizer,'encoder')))
            del encoder,enc_tokenizer,memory,p;gc.collect();torch.cuda.empty_cache()
        update(phase='load_matched_reader',cache_contexts=len(frozen_cache.entries) if frozen_cache else 0,
            cache_bytes=sum(t.numel()*t.element_size() for t in frozen_cache.entries.values()) if frozen_cache else 0)
        model,reader_tokenizer=load_model('full',args.seed);model.eval()
        if tokenizer.encode(' 1234')!=reader_tokenizer.encode(' 1234'):raise ValueError('Tokenizer mismatch')
        if frozen_cache is not None and not torch.equal(model.ft.detach().cpu(),frozen_cache.ft):raise ValueError('FT mismatch')
        for uid,expected in replay:
            actual=greedy(model,prefix(model,items[uid],0,0,frozen_cache,args.arm),tokenizer)
            if actual!=expected:raise RuntimeError('Official encoder/zero reader replay mismatch')
        parameters=[p for p in model.parameters() if p.requires_grad]
        if sum(p.numel() for p in parameters)!=6815744:raise ValueError('Matched reader capacity required')
        initial=state_identity(adapter_state(model))['sha256']
        versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad};ft=model.ft.detach().cpu().clone()
        write_json(output/'config.json',dict(phase=args.phase,arm=args.arm,method=args.method,seed=args.seed,maximum_steps=max_steps,
            schedule=order,initial_reader_identity=initial,trainable_parameters=6815744,source_hashes=source_hashes,
            cache_sha256=cache_digest,cache_persisted_to_disk=False,source_replay_exact=len(replay),
            peak_lr=5e-5 if args.phase=='main' else 1e-4,warmup=32,cosine_floor_fraction=.1,
            natural_loss_weight=.5,probe_loss_weight=.5,assignment_weight=.1,locality_weight=.05,
            encoder_trainable=False,runtime=runtime_settings(),confirmation_loaded=False,reserved_test_accessed=False))
        optimizer=torch.optim.AdamW(parameters,lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        evaluations={};seconds=[];best_key=-math.inf;best_step=None;best_hash=None;passed=args.phase!='overfit'
        torch.cuda.reset_peak_memory_stats();completed=0
        with (output/'steps.jsonl').open('x',encoding='utf8',buffering=1) as f:
            for completed,uid in enumerate(order,1):
                optimizer.zero_grad(set_to_none=True)
                rate=learning_rate(completed,max_steps,5e-5,32) if args.phase=='main' else 1e-4*min(completed/32,1.)
                for group in optimizer.param_groups:group['lr']=rate
                torch.cuda.synchronize();tick=time.perf_counter()
                values=[backward(model,items[u['id']],frozen_cache,args.arm,args.method,.5) for u in by_source[uid]]
                loss=math.fsum(v['loss'] for v in values)
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                if not math.isfinite(loss) or not any(p.grad is not None and bool(p.grad.count_nonzero()) for p in parameters):raise FloatingPointError('Missing/nonfinite gradient')
                optimizer.step();torch.cuda.synchronize();elapsed=time.perf_counter()-tick;seconds.append(elapsed)
                f.write(json.dumps(dict(step=completed,source_id=uid,loss=loss,families=values,gradient_norm=float(norm),learning_rate=rate,seconds=elapsed),allow_nan=False)+'\n')
                if completed==1 or completed%16==0:
                    update(phase='training',optimizer_steps=completed,last_step_seconds=elapsed,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,cumulative_training_seconds=math.fsum(seconds))
                checkpoints=(128,256,512,1024) if args.phase=='overfit' else (128,256,512)
                if completed in checkpoints or completed==max_steps:
                    update(phase='evaluation',optimizer_steps=completed)
                    result=evaluate(model,tokenizer,development,items,frozen_cache,args.arm,output/f'evaluation-{completed}',update)
                    evaluations[str(completed)]=result;write_json(output/'evaluations.json',evaluations)
                    key=selection_key(result)
                    if key>best_key:
                        best_key=key;best_step=completed;save_file(adapter_state(model),str(output/'best-adapter.safetensors'));best_hash=sha(output/'best-adapter.safetensors')
                    if args.phase=='overfit' and gate(result):passed=True;break
        final=adapter_state(model);final_identity=state_identity(final)['sha256'];save_file(final,str(output/'final-adapter.safetensors'))
        first=train_units[0];item=items[first['id']];expected=greedy(model,prefix(model,item,0,0,frozen_cache,args.arm),tokenizer)
        load_state(model,load_file(str(output/'final-adapter.safetensors'),device='cpu'))
        if state_identity(adapter_state(model))['sha256']!=final_identity or greedy(model,prefix(model,item,0,0,frozen_cache,args.arm),tokenizer)!=expected:raise RuntimeError('Reload mismatch')
        if versions!={n:p._version for n,p in model.named_parameters() if not p.requires_grad} or not torch.equal(ft,model.ft.detach().cpu()):raise RuntimeError('Frozen reader changed')
        if frozen_cache is not None and frozen_cache.digest()!=cache_digest:raise RuntimeError('Context-only memory changed')
        if any(sha(ROOT/n)!=h for n,h in source_hashes.items()):raise RuntimeError('Pinned source/data changed')
        summary=dict(status='success' if passed else 'gate_failed',phase=args.phase,arm=args.arm,method=args.method,seed=args.seed,optimizer_steps=completed,
            overfit_pass=passed,evaluations=evaluations,initial_reader_identity=initial,final_reader_identity=final_identity,
            final_checkpoint_sha256=sha(output/'final-adapter.safetensors'),best_development_step=best_step,best_checkpoint_sha256=best_hash,
            best_selection_rule='0.5 macro-level target-joint +0.5 natural knownF1; earlier on ties',
            cache_sha256=cache_digest,cache_unchanged=True,frozen_parameters_unchanged=True,reload_exact=True,
            source_replay_exact=len(replay),cumulative_training_seconds=math.fsum(seconds),peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            writer_cache_peak_mib=writer_peak,wall_seconds=time.perf_counter()-started,confirmation_evaluated=False,reserved_test_accessed=False)
        write_json(output/'summary.json',summary);update(status='success',phase='complete',optimizer_steps=completed,overfit_pass=passed)
        if not passed:raise RuntimeError('Overfit gate not met; do not start main jobs')
        return summary
    except BaseException as error:
        update(status='failed',phase='failed',error=str(error));(output/'failure.txt').write_text(traceback.format_exc(),encoding='utf8');raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','smoke','overfit','main'),required=True)
    parser.add_argument('--arm',choices=('compressed','full'),default='compressed');parser.add_argument('--method',choices=('sft','assignment','relation'),default='sft')
    parser.add_argument('--seed',type=int,choices=(73,97),default=73);args=parser.parse_args()
    if args.phase=='prepare':prepare_data()
    else:train(args)

if __name__=='__main__':main()
