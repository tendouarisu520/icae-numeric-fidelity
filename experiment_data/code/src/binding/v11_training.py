"""Rank16 natural relation memory experiment; remote E only, <=1024 updates."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
from .v11_pipeline import prepare_unit, read_rows, sha, write_json, prediction_rows, metrics

ROOT=Path('E:/Research/ICAE-ConditionalMemory-20261021')
COHORT='data/v11-natural-frozen-20261025'
RUNS='runs/v11-natural-20261025'


def schedule(ids,seed,steps=1024):
    if seed not in (73,97) or not 1<=steps<=1024 or len(ids)!=len(set(ids)) or not ids:
        raise ValueError('Unique units, registered seed and <=1024 updates required')
    rng=random.Random(seed);result=[]
    while len(result)<steps:
        epoch=list(ids);rng.shuffle(epoch);result.extend(epoch)
    return result[:steps]


def overfit_pass(result):
    r=result['rates']
    return result['units']==8 and r['target_all_four']>=7/8 and r['known_exact']>=.89 and r['unknown_exact']>=.9


def load_model(arm,seed):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from safetensors.torch import load_file
    from .model import load_icae
    from .v11_model import configure_residual
    base=str(ROOT/'models/mistral');adapter=str(ROOT/'models/icae/mistral_7b_ft_icae.safetensors')
    if arm=='encoder':
        model,tokenizer=load_icae(base,adapter,device='cuda',dtype=torch.bfloat16)
        configure_residual(model,seed)
    else:
        from .v4_model import Reader
        ft=load_file(adapter,device='cpu')['memory_token_embed.weight'][130]
        backbone=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16,attn_implementation='sdpa',
                       local_files_only=True,low_cpu_mem_usage=True,device_map={'':'cuda'})
        model=Reader(backbone,ft,seed=seed).to('cuda');model.eval()
        tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,use_fast=False)
    model.eval()
    if sum(p.numel() for p in model.parameters() if p.requires_grad)!=6815744:
        raise ValueError('Matched6,815,744 trainable parameters required')
    return model,tokenizer


def prefix(model,prepared,i,j,arm,memory=None):
    query=prepared['queries'][j]
    if arm=='encoder':return model.prefix_ids(memory,query['left'],query['right'])
    return model.prefix(dict(context=prepared['contexts'][i],**query),'full_reader')


def generate(model,inputs,tokenizer,arm):
    import torch
    from .v11_model import frozen_decoder
    generated=[];cache=None
    with torch.no_grad(),(frozen_decoder(model) if arm=='encoder' else nullcontext()):
        for _ in range(32):
            output=(model.base if arm=='encoder' else model.forward_prefix)(
                inputs_embeds=inputs,past_key_values=cache,use_cache=True,return_dict=True)
            token=int(output.logits[0,-1].argmax());generated.append(token)
            if token==tokenizer.eos_token_id:break
            cache=output.past_key_values;inputs=model.embeddings([token]).unsqueeze(0)
    return tokenizer.decode(generated,skip_special_tokens=True),dict(token_ids=generated,eos_emitted=generated[-1]==tokenizer.eos_token_id)


def evaluate(model,tokenizer,units,prepared,arm,output):
    import torch
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[];started=time.perf_counter()
    with torch.no_grad(),(output/'predictions.jsonl').open('x',encoding='utf8',buffering=1) as handle:
        for unit in units:
            item=prepared[unit['id']];predictions=[];generation=[]
            for i in range(4):
                memory=model.compress_ids(torch.tensor([item['contexts'][i]],device=model.device)) if arm=='encoder' else None
                values=[];details=[]
                for j in range(3):
                    prediction,detail=generate(model,prefix(model,item,i,j,arm,memory),tokenizer,arm)
                    values.append(prediction);details.append(detail)
                predictions.append(values);generation.extend(details)
                del memory
            group=prediction_rows(unit,predictions)
            for row,detail in zip(group,generation):
                row.update(generation=detail);handle.write(json.dumps(row,ensure_ascii=False)+'\n')
            rows.extend(group)
    result=metrics(rows);result.update(wall_seconds=time.perf_counter()-started,predictions_sha256=sha(output/'predictions.jsonl'))
    write_json(output/'metrics.json',result);return result


def reader_backward(model,unit):
    import torch
    value=0.;gold=(0,1,2,0)
    for i in range(4):
        for j in range(3):
            answer=unit['candidates'][gold[i] if j==0 else 2] if j!=1 else unit['unaffected_answer']
            loss=model.answer_nll(prefix(model,unit,i,j,'full'),answer)/12
            value+=float(loss.detach());loss.backward()
    return dict(loss=value,sft_loss=value,supervised_answers=12)


def train(args):
    from .repro_probe import configure_environment,configure_torch,runtime_settings
    environment=configure_environment('det_auto')
    import torch
    import numpy as np
    from safetensors.torch import save_file,load_file
    from .v11_model import backward_unit
    from .v9_model import adapter_state,load_state
    from .v4_model import state_identity
    if ROOT.drive.lower()!='e:' or args.arm not in ('encoder','full') or args.method not in ('sft','relation'):
        raise ValueError('Registered remote arms only')
    if (args.arm=='full' and args.method!='sft') or args.seed not in (73,97):raise ValueError('Invalid cell')
    if args.phase!='main' and (args.arm,args.method,args.seed)!=('encoder','relation' if args.phase=='smoke' else 'sft',73):
        raise ValueError('Fixed calibration jobs only')
    name=f'{args.phase}-{args.arm}-{args.method}-seed{args.seed}'
    output=ROOT/RUNS/name
    if str(Path(args.output))!=str(output) or output.exists():raise ValueError('New fixed run path required')
    configure_torch('det_auto');random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    status=dict(status='running',phase='load_model',arm=args.arm,method=args.method,seed=args.seed,
                experiment_phase=args.phase,optimizer_steps=0,pid=os.getpid())
    def update(**values):
        status.update(values,updated=time.time());write_json(output/'status.json',status);print(json.dumps(status),flush=True)
    update()
    try:
        folder=ROOT/COHORT;manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
        # Development is the only held-out partition opened here. Confirmation is not loaded.
        for split in ('train','development'):
            if sha(folder/(split+'.jsonl'))!=manifest['partitions'][split]['sha256']:raise ValueError('Frozen cohort changed')
        train_rows=read_rows(folder/'train.jsonl');development=read_rows(folder/'development.jsonl')
        if len(train_rows)!=512 or len(development)!=32:raise ValueError('Fixed512/32 cohort required')
        by_id={u['id']:u for u in train_rows};overfit=[by_id[uid] for uid in manifest['overfit_ids']]
        units=overfit if args.phase!='main' else train_rows
        steps=2 if args.phase=='smoke' else 1024
        order=schedule([u['id'] for u in units],args.seed,steps)
        model,tokenizer=load_model(args.arm,args.seed)
        prepared={u['id']:prepare_unit(u,tokenizer) for u in train_rows+development}
        trainable=[p for p in model.parameters() if p.requires_grad]
        frozen={name:p for name,p in model.named_parameters() if not p.requires_grad}
        versions={name:p._version for name,p in frozen.items()}
        initial=state_identity(adapter_state(model))
        ft=(model.memory_token_embed.weight if args.arm=='encoder' else model.ft).detach().clone()
        sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/binding/v11_model.py',ROOT/'src/binding/v11_pipeline.py',ROOT/'src/binding/v11_data.py']}
        write_json(output/'config.json',dict(phase=args.phase,arm=args.arm,method=args.method,seed=args.seed,steps=steps,
                   cohort_manifest_sha256=sha(folder/'manifest.json'),source_hashes=sources,
                   initial_adapter_identity=initial,schedule=order,environment=environment,runtime=runtime_settings(),
                   trainable_parameters=sum(p.numel() for p in trainable),learning_rate=1e-4,warmup=32,
                   dropout=False,supervised_rows_per_unit=12,checkpoint_steps=[256,512,1024],
                   confirmation_loaded=False,reserved_test_accessed=False))
        optimizer=torch.optim.AdamW(trainable,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        torch.cuda.reset_peak_memory_stats(model.device);seconds=[];evaluations={};gate=False;completed=0
        with (output/'steps.jsonl').open('x',encoding='utf8',buffering=1) as handle:
            for completed,uid in enumerate(order,1):
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=1e-4*min(completed/32,1.)
                torch.cuda.synchronize();tick=time.perf_counter()
                result=backward_unit(model,prepared[uid],args.method) if args.arm=='encoder' else reader_backward(model,prepared[uid])
                norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
                if not math.isfinite(result['loss']) or not any(p.grad is not None and bool(p.grad.count_nonzero()) for p in trainable):
                    raise FloatingPointError('Nonfinite loss or missing nonzero gradient')
                optimizer.step();torch.cuda.synchronize();elapsed=time.perf_counter()-tick;seconds.append(elapsed)
                if not all(bool(torch.isfinite(p).all()) for p in trainable):raise FloatingPointError('Nonfinite adapter')
                result.update(step=completed,unit_id=uid,gradient_norm=float(norm),learning_rate=optimizer.param_groups[0]['lr'],seconds=elapsed)
                handle.write(json.dumps(result,allow_nan=False)+'\n')
                if completed==1 or completed%16==0:
                    if versions!={name:p._version for name,p in model.named_parameters() if not p.requires_grad}:raise RuntimeError('Frozen parameters changed')
                    update(phase='training',optimizer_steps=completed,last_step_seconds=elapsed,
                           peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,cumulative_training_seconds=math.fsum(seconds))
                evaluate_now=(args.phase=='overfit' and completed in (128,256,512,1024)) or (args.phase=='main' and completed in (256,512,1024))
                if evaluate_now:
                    update(phase='evaluation',optimizer_steps=completed)
                    evaluations[str(completed)]={}
                    for split,values in [('train8',overfit)]+([('development',development)] if args.phase=='main' else []):
                        evaluations[str(completed)][split]=evaluate(model,tokenizer,values,prepared,args.arm,output/f'evaluation-{completed}'/split)
                    save_file(adapter_state(model),str(output/f'adapter-{completed}.safetensors'))
                    write_json(output/'evaluations.json',evaluations)
                    if args.phase=='overfit' and overfit_pass(evaluations[str(completed)]['train8']):gate=True;break
        checkpoint=output/'final-adapter.safetensors';state=adapter_state(model);save_file(state,str(checkpoint))
        reloaded=load_file(str(checkpoint),device='cpu')
        if set(state)!=set(reloaded) or any(not torch.equal(state[k],reloaded[k]) for k in state):raise RuntimeError('Serialized residual differs')
        replay=prepared[overfit[0]['id']]
        with torch.no_grad():
            memory=model.compress_ids(torch.tensor([replay['contexts'][0]],device=model.device)) if args.arm=='encoder' else None
            expected=generate(model,prefix(model,replay,0,0,args.arm,memory),tokenizer,args.arm)
            load_state(model,reloaded)
            memory=model.compress_ids(torch.tensor([replay['contexts'][0]],device=model.device)) if args.arm=='encoder' else None
            actual=generate(model,prefix(model,replay,0,0,args.arm,memory),tokenizer,args.arm)
        if actual!=expected:raise RuntimeError('Checkpoint reload generation differs')
        if versions!={name:p._version for name,p in model.named_parameters() if not p.requires_grad}:raise RuntimeError('Frozen parameter version changed')
        if not torch.equal(ft,model.memory_token_embed.weight if args.arm=='encoder' else model.ft):raise RuntimeError('Frozen FT changed')
        if any(sha(ROOT/name)!=value for name,value in sources.items()):raise RuntimeError('Source changed during run')
        summary=dict(status='success',optimizer_steps=completed,phase=args.phase,arm=args.arm,method=args.method,seed=args.seed,
                     overfit_pass=gate if args.phase=='overfit' else None,evaluations=evaluations,
                     checkpoint_sha256=sha(checkpoint),reload_exact=True,frozen_parameters_unchanged=True,
                     initial_adapter_identity=initial['sha256'],final_adapter_identity=state_identity(state)['sha256'],
                     cumulative_training_seconds=math.fsum(seconds),wall_seconds=time.perf_counter()-started,
                     peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,confirmation_loaded=False,reserved_test_accessed=False)
        write_json(output/'summary.json',summary);update(status='success',phase='complete',optimizer_steps=completed,overfit_pass=summary['overfit_pass'])
        return summary
    except BaseException as error:
        update(status='failed',phase='failed',error=str(error));(output/'failure.txt').write_text(traceback.format_exc(),encoding='utf8');raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('encoder','full'),required=True);parser.add_argument('--method',choices=('sft','relation'),required=True)
    parser.add_argument('--seed',type=int,choices=(73,97),required=True);parser.add_argument('--phase',choices=('smoke','overfit','main'),required=True)
    parser.add_argument('--output',required=True);train(parser.parse_args())


if __name__=='__main__':main()
