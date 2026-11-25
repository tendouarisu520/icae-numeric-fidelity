"""Frozen V11 memories, fresh matched readers; remote E only and <=1024steps."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
from .v11_pipeline import prepare_unit,read_rows,sha,write_json
from .v12_protocol import SOURCES,shuffled_donors

ROOT=Path('E:/Research/ICAE-ConditionalMemory-20261021')
RUNS=ROOT/'runs/v12-frozen-reader-20261025'


def train(args):
    from .repro_probe import configure_environment,configure_torch,runtime_settings
    environment=configure_environment('det_auto')
    import numpy as np
    import torch
    from safetensors.torch import load_file,save_file
    from .v11_training import load_model,schedule
    from .v9_model import load_state,adapter_state
    from .v4_model import state_identity
    from .v12_reader import build_ram_cache,evaluate,require_reproduction,reader_backward,diagnose,prefix_for,greedy
    if ROOT.drive.lower()!='e:' or args.source not in SOURCES or args.seed not in (73,97):raise ValueError('Registered E cells only')
    if args.phase=='smoke' and (args.source,args.seed)!=('sft512',73):raise ValueError('Fixed smoke cell only')
    configure_torch('det_auto');random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    name=f'{args.phase}-{args.source}-seed{args.seed}';output=RUNS/name
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    status=dict(status='running',phase='preflight',job=name,pid=os.getpid(),optimizer_steps=0)
    def update(**values):
        status.update(values,updated=time.time());write_json(output/'status.json',status);print(json.dumps(status),flush=True)
    update()
    try:
        oldruns=ROOT/'runs/v11-natural-20261025'
        if json.loads((oldruns/'queue-status.json').read_text(encoding='utf8'))['status']!='success':raise ValueError('Complete V11 first')
        cohort=ROOT/'data/v11-natural-frozen-20261025';manifest=json.loads((cohort/'manifest.json').read_text(encoding='utf8'))
        for split in ('train','development'):
            if sha(cohort/(split+'.jsonl'))!=manifest['partitions'][split]['sha256']:raise ValueError('Frozen cohort changed')
        train_rows=read_rows(cohort/'train.jsonl');dev=read_rows(cohort/'development.jsonl')
        if (len(train_rows),len(dev))!=(512,32):raise ValueError('Fixed512/32 required')
        by_id={u['id']:u for u in train_rows};overfit=[by_id[k] for k in manifest['overfit_ids']]
        units=overfit[:2] if args.phase=='smoke' else train_rows
        development=dev[:2] if args.phase=='smoke' else dev
        method,source_step=SOURCES[args.source]
        source_folder=oldruns/f'main-encoder-{method}-seed{args.seed}';source_checkpoint=source_folder/f'adapter-{source_step}.safetensors'
        full_folder=oldruns/f'main-full-sft-seed{args.seed}';full_checkpoint=full_folder/f'adapter-{source_step}.safetensors'
        source_summary=json.loads((source_folder/'summary.json').read_text(encoding='utf8'))
        full_summary=json.loads((full_folder/'summary.json').read_text(encoding='utf8'))
        if any(s['status']!='success' or s['optimizer_steps']!=1024 for s in (source_summary,full_summary)):
            raise ValueError('Incomplete parent comparison')
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (source_checkpoint,full_checkpoint,
            ROOT/'src/binding/v12_training.py',ROOT/'src/binding/v12_reader.py',ROOT/'src/binding/v12_protocol.py',
            ROOT/'src/binding/v11_training.py',ROOT/'src/binding/v11_pipeline.py',ROOT/'src/binding/v4_model.py')}
        update(phase='load_frozen_encoder');torch.cuda.reset_peak_memory_stats()
        encoder,tokenizer=load_model('encoder',args.seed)
        source_state=load_file(str(source_checkpoint),device='cpu');load_state(encoder,source_state)
        writer_identity=state_identity(source_state)['sha256'];del source_state
        encoder.requires_grad_(False);encoder.eval()
        prepared={u['id']:prepare_unit(u,tokenizer) for u in units+overfit+development}
        cache_units=units+overfit+development
        contexts=[ids for u in cache_units for ids in prepared[u['id']]['contexts']]
        encoder_versions={n:p._version for n,p in encoder.named_parameters()}
        update(phase='encode_frozen_memories')
        cache=build_ram_cache(encoder,contexts,update);cache_identity=cache.digest()
        if encoder_versions!={n:p._version for n,p in encoder.named_parameters()}:raise RuntimeError('Frozen encoder changed')
        writer_peak=torch.cuda.max_memory_allocated()/1024**2
        del encoder;gc.collect();torch.cuda.empty_cache()
        update(phase='load_matched_reader',cache_contexts=len(cache.entries),cache_bytes=sum(t.numel()*t.element_size() for t in cache.entries.values()))
        model,reader_tokenizer=load_model('full',args.seed);model.eval()
        if tokenizer.encode(' UNKNOWN',add_special_tokens=False)!=reader_tokenizer.encode(' UNKNOWN',add_special_tokens=False):
            raise ValueError('Tokenizer mismatch')
        if not torch.equal(cache.ft,model.ft.detach().cpu()):raise ValueError('Memory FT delimiter differs')
        initial=adapter_state(model);initial_identity=state_identity(initial)['sha256']
        if initial_identity!=full_summary['initial_adapter_identity']:raise ValueError('Reader initialization differs from full control')
        trainable=[p for p in model.parameters() if p.requires_grad]
        if sum(p.numel() for p in trainable)!=6815744:raise ValueError('Matched6,815,744parameters required')
        frozen_versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
        ft=model.ft.detach().cpu().clone()
        order=schedule([u['id'] for u in units],args.seed,2 if args.phase=='smoke' else 1024)
        if args.phase=='main':
            full_config=json.loads((full_folder/'config.json').read_text(encoding='utf8'))
            if order!=full_config['schedule']:raise ValueError('Training exposure differs from full comparison')
        write_json(output/'config.json',dict(source=args.source,seed=args.seed,phase=args.phase,steps=len(order),schedule=order,
            source_hashes=source_hashes,source_step=source_step,writer_adapter_identity=writer_identity,cache_sha256=cache_identity,
            cache_contexts=len(cache.entries),cache_bytes=status['cache_bytes'],cache_persisted_to_disk=False,
            initial_reader_identity=initial_identity,trainable_parameters=6815744,learning_rate=1e-4,warmup=32,dropout=False,
            environment=environment,runtime=runtime_settings(),cohort_manifest_sha256=sha(cohort/'manifest.json'),
            confirmation_loaded=False,reserved_test_accessed=False))
        update(phase='reproduce_source_predictions')
        baseline,baseline_rows=evaluate(model,tokenizer,development,prepared,cache,output/'baseline-matched')
        expected_all=read_rows(source_folder/f'evaluation-{source_step}/development/predictions.jsonl')
        dev_ids={u['id'] for u in development};expected=[r for r in expected_all if r['unit_id'] in dev_ids]
        replay=require_reproduction(baseline_rows,expected)
        diagnostics={};common=None
        if args.phase=='main':
            update(phase='baseline_unknown_scores')
            diagnostics['baseline_unknown'],common=diagnose(model,tokenizer,development,prepared,cache,baseline_rows,output/'baseline-unknown')
        optimizer=torch.optim.AdamW(trainable,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        torch.cuda.reset_peak_memory_stats();seconds=[];evaluations={};checkpoints={}
        with (output/'steps.jsonl').open('x',encoding='utf8',buffering=1) as handle:
            for step,uid in enumerate(order,1):
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=1e-4*min(step/32,1.)
                torch.cuda.synchronize();tick=time.perf_counter()
                value=reader_backward(model,prepared[uid],cache)
                norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
                if not math.isfinite(value['loss']) or not any(p.grad is not None and bool(p.grad.count_nonzero()) for p in trainable):
                    raise FloatingPointError('Missing/nonfinite reader gradient')
                optimizer.step();torch.cuda.synchronize();elapsed=time.perf_counter()-tick;seconds.append(elapsed)
                if not all(bool(torch.isfinite(p).all()) for p in trainable):raise FloatingPointError('Nonfinite reader adapter')
                value.update(step=step,unit_id=uid,gradient_norm=float(norm),learning_rate=optimizer.param_groups[0]['lr'],seconds=elapsed)
                handle.write(json.dumps(value,allow_nan=False)+'\n')
                if step==1 or step%16==0:
                    if frozen_versions!={n:p._version for n,p in model.named_parameters() if not p.requires_grad}:raise RuntimeError('Frozen reader base changed')
                    update(phase='training',optimizer_steps=step,last_step_seconds=elapsed,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                           cumulative_training_seconds=math.fsum(seconds))
                if step in (256,512,1024) or step==len(order):
                    update(phase='evaluation',optimizer_steps=step)
                    evaluation,final_rows=evaluate(model,tokenizer,development,prepared,cache,output/f'evaluation-{step}/development')
                    train_eval,_=evaluate(model,tokenizer,overfit,prepared,cache,output/f'evaluation-{step}/train8')
                    evaluations[str(step)]=dict(development=evaluation,train8=train_eval)
                    path=output/f'adapter-{step}.safetensors';save_file(adapter_state(model),str(path));checkpoints[str(step)]=sha(path)
                    write_json(output/'evaluations.json',evaluations)
        final_state=adapter_state(model);final_identity=state_identity(final_state)['sha256']
        if args.phase=='main':
            update(phase='trained_unknown_scores',optimizer_steps=1024)
            diagnostics['trained_unknown'],_=diagnose(model,tokenizer,development,prepared,cache,final_rows,output/'trained-unknown',common=common)
            donors=shuffled_donors(development)
            for mode in ('shuffled','null'):
                update(phase='intervention_'+mode)
                diagnostics[mode],_=evaluate(model,tokenizer,development,prepared,cache,output/('trained-'+mode),mode,donors if mode=='shuffled' else None)
            update(phase='matched_full_reader_scores')
            load_state(model,load_file(str(full_checkpoint),device='cpu'))
            full_result,full_rows=evaluate(model,tokenizer,development,prepared,cache,output/'matched-full-reader',mode='full')
            full_expected=read_rows(full_folder/f'evaluation-{source_step}/development/predictions.jsonl')
            full_replay=require_reproduction(full_rows,full_expected)
            diagnostics['full_reader'],_=diagnose(model,tokenizer,development,prepared,cache,full_rows,output/'full-unknown',mode='full',common=common)
            diagnostics['full_reader_metrics']=full_result;diagnostics['full_replay']=full_replay
        load_state(model,load_file(str(output/f'adapter-{len(order)}.safetensors'),device='cpu'))
        if state_identity(adapter_state(model))['sha256']!=final_identity:raise RuntimeError('Reloaded reader differs')
        first=development[0];actual=greedy(model,prefix_for(model,prepared[first['id']],0,0,cache),tokenizer)
        if actual[0]!=final_rows[0]['prediction'] or actual[1]!=final_rows[0]['generation']:raise RuntimeError('Reload generation differs')
        if cache.digest()!=cache_identity:raise RuntimeError('Frozen memories changed')
        if frozen_versions!={n:p._version for n,p in model.named_parameters() if not p.requires_grad} or not torch.equal(ft,model.ft.detach().cpu()):
            raise RuntimeError('Frozen reader parameters changed')
        if any(sha(ROOT/p)!=h for p,h in source_hashes.items()):raise RuntimeError('Pinned source/checkpoint changed')
        summary=dict(status='success',source=args.source,seed=args.seed,phase=args.phase,optimizer_steps=len(order),
            baseline=baseline,source_reproduction=replay,evaluations=evaluations,diagnostics=diagnostics,
            checkpoints=checkpoints,initial_reader_identity=initial_identity,final_reader_identity=final_identity,
            cache_sha256=cache_identity,cache_unchanged=True,source_checkpoints_unchanged=True,frozen_parameters_unchanged=True,reload_exact=True,
            cumulative_training_seconds=math.fsum(seconds),peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            writer_cache_peak_mib=writer_peak,wall_seconds=time.perf_counter()-started,confirmation_evaluated=False,reserved_test_accessed=False)
        write_json(output/'summary.json',summary);update(status='success',phase='complete',optimizer_steps=len(order));return summary
    except BaseException as error:
        update(status='failed',phase='failed',error=str(error));(output/'failure.txt').write_text(traceback.format_exc(),encoding='utf8');raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',choices=tuple(SOURCES),required=True)
    parser.add_argument('--seed',type=int,choices=(73,97),required=True);parser.add_argument('--phase',choices=('smoke','main'),required=True)
    train(parser.parse_args())


if __name__=='__main__':main()
