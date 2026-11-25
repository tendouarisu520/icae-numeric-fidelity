"""V4 bounded matched-reader diagnostic. Fresh CLI sets determinism before Torch.

cache and train are separate processes: the official encoder is released before
either reader is loaded. No old dev/test loader, resume, search or test CLI.
"""
import argparse
import collections
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback

from .paired_data import LEVELS, canonical_hash
from .repro_probe import configure_environment, configure_torch, runtime_settings
from .v4_data import SPLITS, load_suite, overfit_rows

ARMS=('full_reader','memory_reader','encoder_sft')
STAGES=('overfit','main')
MAX_NEW_TOKENS=32
SOURCES=('__init__.py','data.py','paired_data.py','model.py','objectives.py','benchmark.py',
    'metrics.py','eval_metrics.py','repro_probe.py','paired_training.py',
    'v4_data.py','v4_model.py','v4_training.py')


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


@dataclass
class TrainConfig:
    suite:str
    base_path:str
    adapter_path:str
    output:str
    cache:str
    profile:str
    arm:str='full_reader'
    stage:str='overfit'
    seed:int=73
    steps:int=1024
    device:str='cuda'
    smoke:bool=False


def write_json(path,value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def validate_config(config,fixture=False):
    if config.arm not in ARMS or config.stage not in STAGES or config.profile!='det_auto':
        raise ValueError('Require valid arm/stage and strict det_auto profile')
    if type(config.steps)!=int or not 1<=config.steps<=1024:
        raise ValueError('Training step cap is 1024')
    if config.arm=='encoder_sft' and config.stage!='overfit':
        raise ValueError('encoder_sft is an overfit-only anchor')
    if config.seed!=73:
        raise ValueError('Frozen seed must be73')
    if config.stage=='main' and config.steps!=1024:
        raise ValueError('Main stage requires exactly1024 updates')
    if config.smoke and (config.stage!='overfit' or config.steps not in (1,2)):
        raise ValueError('Smoke is overfit-only and limited to1or2 updates')
    if not fixture and not config.smoke and config.steps!=1024:
        raise ValueError('Production starts with the frozen1024 maximum; only overfit gate may stop earlier')


def build_schedule(rows,seed,steps):
    if type(steps)!=int or not 1<=steps<=1024 or not rows:
        raise ValueError('Schedule steps must be1..1024')
    groups=collections.defaultdict(list)
    seen=set()
    for row in rows:
        if row['id'] in seen or row['split']!='train':
            raise ValueError('Schedule requires unique train-only rows')
        seen.add(row['id']); groups[row['scenario_id']].append(row)
    for group in groups.values():
        pairs=collections.Counter(r['pair_id'] for r in group if r['kind']=='swap')
        if len(group)!=5 or sum(r['kind']=='unknown' for r in group)!=1 or len(pairs)!=2 or set(pairs.values())!={2}:
            raise ValueError('Schedule requires complete five-row scenarios')
    scenarios=sorted(groups); schedule=[]; epoch=0
    while len(schedule)<steps:
        shuffled=list(scenarios)
        random.Random(canonical_hash([seed,'v4_all_scenarios',epoch])).shuffle(shuffled)
        for sid in shuffled:
            index=len(schedule)
            ids=sorted(r['id'] for r in groups[sid])
            random.Random(canonical_hash([seed,'v4_row_order',index,sid])).shuffle(ids)
            schedule.append(dict(unit_index=index,scenario_id=sid,row_ids=ids,epoch=epoch))
            if len(schedule)==steps: break
        epoch+=1
    return schedule


def summarize(rows,predictions):
    from .metrics import evaluate_predictions,normalize,_index
    from .eval_metrics import diagnose_output
    indexed=_index(rows,predictions)
    if len(indexed)!=len(rows): raise ValueError('Complete raw predictions required')
    result=evaluate_predictions(rows,predictions)
    known=[r for r in rows if r['kind']=='swap']; unknown=[r for r in rows if r['kind']=='unknown']
    if not known or not unknown: raise ValueError('Known and UNKNOWN rows required')
    correct=lambda r:normalize(indexed[r['id']]['prediction'])==normalize(r['answer'])
    formatted=lambda r:diagnose_output(indexed[r['id']]['prediction'],
        indexed[r['id']].get('generation',{}).get('stop_reason')=='max_new_tokens')['category']=='exact_single_label'
    result.update(known_rows=len(known),unknown_rows=len(unknown),scenarios=len({r['scenario_id'] for r in rows}),
        known_correct=sum(correct(r) for r in known),unknown_correct=sum(correct(r) for r in unknown),
        known_accuracy=sum(correct(r) for r in known)/len(known),
        known_format_rate=sum(formatted(r) for r in known)/len(known),
        unknown_format_rate=sum(formatted(r) for r in unknown)/len(unknown),
        foil_misbinding_rate=sum(normalize(indexed[r['id']]['prediction'])==normalize(r['foil']) for r in known)/len(known),
        false_unknown_rate=sum(normalize(indexed[r['id']]['prediction'])=='unknown' for r in known)/len(known))
    return result


def overfit_gate(metrics,finite):
    return bool(finite) and all(metrics[k]>=.95 for k in ('known_accuracy','unknown_accuracy','pair_accuracy')) and all(
        metrics[k]==1. for k in ('known_format_rate','unknown_format_rate'))


def selection_key(step,evaluation):
    metrics=evaluation['val_id']['pooled']
    values=[metrics[k] for k in ('known_accuracy','unknown_accuracy','pair_accuracy')]
    if any(not math.isfinite(v) or not 0<=v<=1 for v in values):
        raise ValueError('Invalid val_id checkpoint metrics')
    return ((values[0]+values[1])/2,values[2],-int(step))


def select_step(evaluations):
    candidates=[step for step in (256,512,1024) if str(step) in evaluations]
    if not candidates: raise ValueError('No eligible val_id selection checkpoint')
    try: return max(candidates,key=lambda step:selection_key(step,evaluations[str(step)]))
    except (KeyError,TypeError) as exc: raise ValueError('Selection requires val_id only') from exc


@contextmanager
def evaluation_state(model):
    import numpy as np
    import torch
    from .v4_model import state_identity
    modes=[(module,module.training) for module in model.modules()]
    py,npstate,cpu=random.getstate(),np.random.get_state(),torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    before=state_identity({n:p for n,p in model.named_parameters() if p.requires_grad})
    versions={n:p._version for n,p in model.named_parameters()}
    try:
        model.eval()
        with torch.no_grad(): yield
    finally:
        after=state_identity({n:p for n,p in model.named_parameters() if p.requires_grad})
        changed=before!=after or versions!={n:p._version for n,p in model.named_parameters()}
        for module,mode in modes: module.training=mode
        random.setstate(py); np.random.set_state(npstate); torch.set_rng_state(cpu)
        if cuda is not None: torch.cuda.set_rng_state_all(cuda)
        if changed: raise RuntimeError('Evaluation changed parameters')


def prefix_for(model,item,arm,cache):
    import torch
    if arm in ('full_reader','memory_reader'):
        return model.prefix(item,arm,cache.get(item['context']) if arm=='memory_reader' else None)
    if arm=='encoder_sft':
        memory=model.compress_ids(torch.tensor([item['context']],device=model.device,dtype=torch.long))
        return model.prefix_ids(memory,item['left'],item['right'])
    raise ValueError('Unknown model arm')


def answer_nll(model,prefix,answer,arm):
    import torch
    if arm in ('full_reader','memory_reader'): return model.answer_nll(prefix,answer)
    if not answer: raise ValueError('Empty answer')
    if prefix.shape[1]+len(answer)>model.base.config.max_position_embeddings:
        raise ValueError('Positional limit exceeded; truncation forbidden')
    inputs=torch.cat((prefix,model.embeddings(answer).unsqueeze(0)),dim=1)
    with model.icae.disable_adapter():
        output=model.base(inputs_embeds=inputs,use_cache=False,return_dict=True)
    logits=output.logits[0,prefix.shape[1]-1:prefix.shape[1]-1+len(answer)].float()
    return torch.nn.functional.cross_entropy(logits,torch.tensor(answer,device=model.device))


def greedy_generate(model,prefix,eos_id,arm,max_new_tokens=MAX_NEW_TOKENS):
    import torch
    if arm not in ARMS or max_new_tokens!=MAX_NEW_TOKENS:
        raise ValueError('Frozen greedy protocol requires valid arm and32 tokens')
    if prefix.shape[1]+max_new_tokens>model.base.config.max_position_embeddings:
        raise ValueError('Generation positional limit exceeded; truncation forbidden')
    generated=[]; kv=None; inputs=prefix
    with torch.no_grad(),(model.icae.disable_adapter() if arm=='encoder_sft' else nullcontext()):
        for _ in range(max_new_tokens):
            output=(model.base(inputs_embeds=inputs,past_key_values=kv,use_cache=True,return_dict=True)
                if arm=='encoder_sft' else model.forward_prefix(inputs,past_key_values=kv,use_cache=True,return_dict=True))
            token=int(output.logits[0,-1].argmax()); generated.append(token)
            if token==eos_id:
                return dict(token_ids=generated,stop_reason='eos',eos_emitted=True,generated_token_count=len(generated))
            kv=output.past_key_values; inputs=model.embeddings([token]).unsqueeze(0)
    return dict(token_ids=generated,stop_reason='max_new_tokens',eos_emitted=False,generated_token_count=len(generated))


def evaluate(model,tokenizer,rows,prepared,arm,cache,output):
    from .eval_metrics import diagnose_output
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    predictions=[]; start=time.perf_counter()
    with evaluation_state(model):
        with (output/'predictions.jsonl').open('x',encoding='utf-8',buffering=1) as handle:
            for row in rows:
                item=prepared[row['id']]; prefix=prefix_for(model,item,arm,cache)
                nll=float(answer_nll(model,prefix,item['answer'],arm))
                if not math.isfinite(nll): raise FloatingPointError('Nonfinite evaluation answer/EOS NLL')
                generation=greedy_generate(model,prefix,tokenizer.eos_token_id,arm)
                decoded=tokenizer.decode(generation['token_ids'],skip_special_tokens=True)
                record=dict(id=row['id'],scenario_id=row['scenario_id'],arm=arm,prediction=decoded,
                    raw_output=tokenizer.decode(generation['token_ids'],skip_special_tokens=False),generation=generation,
                    format=diagnose_output(decoded,generation['stop_reason']=='max_new_tokens'),
                    answer_eos_nll=nll,answer_eos_tokens=len(item['answer']),context_token_count=len(item['context']),
                    prefix_token_count=int(prefix.shape[1]))
                handle.write(json.dumps(record,allow_nan=False)+'\n'); predictions.append(record)
    summary=dict(pooled=summarize(rows,predictions),by_level={},evaluation_state_preserved=True,
        answer_eos_nll_mean=math.fsum(p['answer_eos_nll'] for p in predictions)/len(predictions),
        prefix_token_count_mean=sum(p['prefix_token_count'] for p in predictions)/len(predictions),
        generation_seconds=time.perf_counter()-start,predictions_sha256=sha256_file(output/'predictions.jsonl'))
    for level in sorted({r['difficulty'] for r in rows}):
        chosen=[r for r in rows if r['difficulty']==level]; ids={r['id'] for r in chosen}
        summary['by_level'][f'd{level}']=summarize(chosen,[p for p in predictions if p['id'] in ids])
    write_json(output/'summary.json',summary)
    return summary


def train_unit(model,items,optimizer,arm,cache,seed,unit_index):
    import torch
    from .paired_training import seed_sample
    if arm not in ARMS or len(items)!=5 or len({i['id'] for i in items})!=5 or sorted(i['kind'] for i in items)!=['swap']*4+['unknown']:
        raise ValueError('A complete four-known/one-UNKNOWN five-row mean is required')
    trainables=[p for p in model.parameters() if p.requires_grad]
    if model.device.type=='cuda': torch.cuda.synchronize(model.device)
    start=time.perf_counter(); optimizer.zero_grad(set_to_none=True); losses=[]
    for item in items:
        seed_sample(seed,unit_index,item['id'])
        prefix=prefix_for(model,item,arm,cache)
        loss=answer_nll(model,prefix,item['answer'],arm)
        value=float(loss.detach())
        if not math.isfinite(value): raise FloatingPointError('Nonfinite training loss')
        (loss/5).backward(); losses.append(value)
        del prefix,loss
    if not any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in trainables):
        raise FloatingPointError('No nonzero trainable gradient; reader may be disabled')
    norm=torch.nn.utils.clip_grad_norm_(trainables,1.,error_if_nonfinite=True)
    if not math.isfinite(float(norm)) or any(not bool(torch.isfinite(p.grad).all()) for p in trainables if p.grad is not None):
        raise FloatingPointError('Nonfinite gradients')
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in trainables):
        raise FloatingPointError('Nonfinite parameters after optimizer update')
    if model.device.type=='cuda': torch.cuda.synchronize(model.device)
    return dict(optimizer_step=unit_index+1,row_ids=[i['id'] for i in items],samples_this_step=5,
        row_divisors=[5]*5,row_losses=losses,loss=math.fsum(losses)/5,finite_gradients=True,
        gradient_norm_before_clip=float(norm),training_seconds=time.perf_counter()-start)


class InputBinding:
    def __init__(self,suite,base_path,adapter_path,cache=None):
        self.paths={f'source/{n}':Path(__file__).with_name(n) for n in SOURCES}
        suite=Path(suite)
        self.paths.update({f'suite/{n}':suite/n for n in ('manifest.json','semantic_audit.json',*(f'{s}.jsonl' for s in SPLITS))})
        base=Path(base_path)
        files=[base] if base.is_file() else sorted(p for p in base.rglob('*') if p.is_file())
        if not files: raise ValueError('Base directory absent or empty')
        self.paths.update({f'base/{p.relative_to(base) if base.is_dir() else p.name}':p for p in files})
        self.paths['official_adapter']=Path(adapter_path)
        if cache:
            root=Path(cache)
            self.paths.update({f'cache/{p.name}':p for p in sorted(root.iterdir()) if p.is_file()})
        self.hashes={name:sha256_file(path) for name,path in self.paths.items()}
        self.stats={name:(p.stat().st_size,p.stat().st_mtime_ns) for name,p in self.paths.items()}

    def assert_unchanged(self,full=False):
        for name,path in self.paths.items():
            stat=path.stat()
            if (stat.st_size,stat.st_mtime_ns)!=self.stats[name]: raise ValueError('Input changed: '+name)
            if full or stat.st_size<8*1024*1024:
                if sha256_file(path)!=self.hashes[name]: raise ValueError('Input hash changed: '+name)


def cache_provenance(binding):
    return dict(protocol=4,input_hashes={k:v for k,v in binding.hashes.items() if not k.startswith('cache/')},
        tokenization='frozen benchmark.prepare_row context only; BOS included; max512; no truncation')


def run_cache(args,model=None,tokenizer=None):
    if args.profile!='det_auto': raise ValueError('Require det_auto')
    environment=configure_environment(args.profile)
    import torch
    from .benchmark import prepare_row
    from .model import load_icae
    from .v4_model import build_cache
    target=torch.device(model.device if model is not None else args.device)
    if target.type=='cuda' and environment['torch_preimported'] and torch.cuda.is_initialized():
        raise RuntimeError('Fresh process required before CUDA initialization')
    configure_torch(args.profile)
    suite=load_suite(args.suite); binding=InputBinding(args.suite,args.base_path,args.adapter_path)
    if model is None: model,tokenizer=load_icae(args.base_path,args.adapter_path,device=args.device,dtype=torch.bfloat16)
    if tokenizer is None: raise ValueError('Tokenizer required')
    # Match the anchor's FP32 encoder LoRA + learned memory, BF16 base policy.
    model.retrieval_proj.requires_grad_(False)
    for parameter in model.parameters():
        if parameter.requires_grad: parameter.data=parameter.data.float()
    model.requires_grad_(False)
    contexts=[prepare_row(row,tokenizer,512)['context'] for split in SPLITS for row in suite[split]]
    manifest=build_cache(model,contexts,args.output,cache_provenance(binding))
    binding.assert_unchanged(full=True)
    del model
    gc.collect()
    if target.type=='cuda': torch.cuda.empty_cache()
    return manifest


def run_train(config,model=None,tokenizer=None,evaluate_fn=None):
    output=Path(config.output); output.mkdir(parents=True,exist_ok=False)
    status=dict(status='running',phase='validation',optimizer_steps=0,samples=0)
    def update(**values):
        status.update(values); status['timestamp_unix']=time.time()
        write_json(output/'status.json',status); print(json.dumps(status,allow_nan=False),flush=True)
    update()
    environment=configure_environment(config.profile)
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from .benchmark import prepare_row,_cuda_memory,_gpu_metadata,_versions
    from .model import load_icae
    from .paired_training import optimizer_settings
    from .v4_model import MemoryCache,load_reader,state_identity
    started=time.perf_counter(); fixture=model is not None
    metadata=dict(protocol_version=4,arguments=asdict(config),environment=environment,injected_model=fixture,
        task='bounded training-versus-compressed-reading diagnosis',max_new_tokens=MAX_NEW_TOKENS,
        test_policy='Only V4 train/val_id/val_ood opened. Old test not inspected.',
        full_reader_interface='Matched diagnostic control uses identical frozen official FT delimiter',
        compute_policy='Input lengths and actual resources reported; no equal-FLOP claim')
    write_json(output/'config.json',metadata)
    try:
        validate_config(config,fixture=fixture)
        target=torch.device(model.device if fixture else config.device)
        if target.type=='cuda' and environment['torch_preimported'] and torch.cuda.is_initialized():
            raise RuntimeError('Deterministic CUDA run requires fresh CLI before Torch import')
        configure_torch(config.profile)
        suite=load_suite(config.suite); binding=InputBinding(config.suite,config.base_path,config.adapter_path,config.cache)
        cache=MemoryCache(config.cache,cache_provenance(binding))
        random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
        update(phase='load_model')
        if not fixture:
            if config.arm=='encoder_sft':
                model,tokenizer=load_icae(config.base_path,config.adapter_path,device=config.device,dtype=torch.bfloat16)
            else: model,tokenizer=load_reader(config.base_path,cache,config.seed,config.device)
        if tokenizer is None: raise ValueError('Tokenizer required')
        if config.arm=='encoder_sft':
            model.retrieval_proj.requires_grad_(False)
        else:
            from .v4_model import Reader
            if not isinstance(model,Reader): raise ValueError('Reader arms require active rank16 Reader')
            if model.base.config.hidden_size!=cache.manifest['hidden_size']:
                raise ValueError('Reader/cache hidden size mismatch')
        for p in model.parameters():
            if p.requires_grad: p.data=p.data.float()
        if not fixture and (model.base.get_input_embeddings().weight.dtype!=torch.bfloat16
            or (config.arm=='encoder_sft' and model.memory_size!=128)):
            raise ValueError('Production requires BF16 base and128 ICAE slots')
        if model.base.is_gradient_checkpointing: raise ValueError('Checkpointing forbidden')
        prepared={row['id']:prepare_row(row,tokenizer,512) for split in SPLITS for row in suite[split]}
        cache.validate_contexts([i['context'] for i in prepared.values()])
        train=overfit_rows(suite['train']) if config.stage=='overfit' else suite['train']
        schedule=build_schedule(train,config.seed,config.steps)
        write_json(output/'schedule.json',schedule)
        trainables=[p for p in model.parameters() if p.requires_grad]
        state=(lambda:model.adapter_state(include_projection=False)) if config.arm=='encoder_sft' else model.adapter_state
        initial=state_identity(state()); rate=1e-5 if config.arm=='encoder_sft' else 1e-4
        optimizer=torch.optim.AdamW(trainables,**optimizer_settings(rate))
        model.train(); evaluator=evaluate if evaluate_fn is None else evaluate_fn
        metadata.update(input_hashes=binding.hashes,versions=_versions(),runtime=runtime_settings(),gpu=_gpu_metadata(model.device),
            initial_adapter_identity=initial,trainable_names=sorted(n for n,p in model.named_parameters() if p.requires_grad),
            trainable_parameters=sum(p.numel() for p in trainables),actual_base_dtype=str(model.base.get_input_embeddings().weight.dtype),
            trainable_dtypes=sorted({str(p.dtype) for p in trainables}),optimizer=optimizer_settings(rate),
            row_mean='ordinary5-row mean: 4known+1UNKNOWN; divisor5 each',max_grad_norm=1.,
            schedule_sha256=canonical_hash(schedule),scenario_order_sha256=canonical_hash([u['scenario_id'] for u in schedule]),
            row_order_sha256=canonical_hash([u['row_ids'] for u in schedule]),prepared_sha256=canonical_hash(prepared),
            context_token_max=max(len(i['context']) for i in prepared.values()),
            reader_lora=dict(r=16,alpha=32,target_modules=['q_proj','v_proj'],dropout=.05) if config.arm!='encoder_sft' else None,
            selection='val_id balanced known/UNKNOWN accuracy, then pair accuracy, then earlier step; candidates256,512,1024')
        metadata['config_sha256']=canonical_hash(metadata); write_json(output/'config.json',metadata)
        checkpoints=[] if config.smoke else ([0,256,512,1024] if config.stage=='main' else [64,256,512,1024])
        if fixture and not config.smoke and config.steps not in checkpoints: checkpoints=sorted(set([s for s in checkpoints if s<=config.steps]+[config.steps]))
        evaluations={}; seconds=[]; selected=None; stopped=False; completed=0
        if model.device.type=='cuda': torch.cuda.reset_peak_memory_stats(model.device)
        def eval_step(step):
            nonlocal selected,stopped
            binding.assert_unchanged()
            update(phase='evaluation',evaluation_step=step)
            splits={'train_d12':train} if config.stage=='overfit' else {'val_id':suite['val_id'],'val_ood':suite['val_ood']}
            if config.stage=='main' and step==1024: splits['train_d12']=overfit_rows(suite['train'])
            results={}
            for split,rows in splits.items():
                results[split]=evaluator(model,tokenizer,rows,prepared,config.arm,cache,output/f'evaluation-step-{step:04d}'/split)
            evaluations[str(step)]=results
            write_json(output/'evaluations.json',evaluations)
            if config.stage=='main' and step in (256,512,1024):
                best=select_step(evaluations)
                if best!=selected:
                    selected=best; save_file(state(),str(output/'selected-adapter.safetensors'))
            elif config.stage=='overfit': stopped=overfit_gate(results['train_d12']['pooled'],True)
            binding.assert_unchanged()
        if 0 in checkpoints: eval_step(0)
        with (output/'steps.jsonl').open('x',encoding='utf-8',buffering=1) as log:
            for unit in schedule:
                record=train_unit(model,[prepared[rid] for rid in unit['row_ids']],optimizer,config.arm,cache,config.seed,unit['unit_index'])
                seconds.append(record['training_seconds']); completed=unit['unit_index']+1
                record.update(scenario_id=unit['scenario_id'],epoch=unit['epoch'],samples=completed*5,
                    cumulative_training_seconds=math.fsum(seconds),**_cuda_memory(model.device))
                log.write(json.dumps(record,allow_nan=False)+'\n')
                update(phase='training',optimizer_steps=completed,samples=completed*5,
                    cumulative_training_seconds=math.fsum(seconds),elapsed_seconds=time.perf_counter()-started)
                if completed in checkpoints:
                    eval_step(completed)
                    if config.stage=='overfit' and stopped: break
        update(phase='save_adapters')
        save_file(state(),str(output/'final-adapter.safetensors'))
        if config.stage=='overfit':
            selected=completed; save_file(state(),str(output/'selected-adapter.safetensors'))
        if config.stage=='main' and (completed!=1024 or selected!=select_step(evaluations)):
            raise RuntimeError('Main1024 endpoint/selection invariant failed')
        binding.assert_unchanged(full=True)
        result=dict(status='success',arm=config.arm,stage=config.stage,optimizer_steps=completed,samples=completed*5,
            early_stopped=stopped,selected_step=selected,final_step=completed,evaluations=evaluations,finite_gradients=True,
            initial_adapter_identity=initial,final_adapter_identity=state_identity(state()),
            selected_adapter_sha256=sha256_file(output/'selected-adapter.safetensors'),
            final_adapter_sha256=sha256_file(output/'final-adapter.safetensors'),config_sha256=metadata['config_sha256'],
            input_hashes=binding.hashes,schedule_sha256=metadata['schedule_sha256'],
            cumulative_training_seconds=math.fsum(seconds),training_step_seconds=seconds,
            run_wall_seconds=time.perf_counter()-started,**_cuda_memory(model.device),
            interpretation='Single-seed bounded synthetic diagnostic; no condition-binding efficacy claim')
        write_json(output/'summary.json',result); update(status='success',phase='complete')
        return result
    except BaseException as exc:
        update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        write_json(output/'summary.json',status)
        (output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise


def argument_parser():
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest='command',required=True)
    for name in ('cache','train'):
        sub=commands.add_parser(name)
        for field in ('suite','base-path','adapter-path','output'): sub.add_argument('--'+field,required=True)
        sub.add_argument('--profile',choices=('det_auto',),required=True); sub.add_argument('--device',default='cuda')
        if name=='train':
            sub.add_argument('--cache',required=True); sub.add_argument('--arm',choices=ARMS,required=True)
            sub.add_argument('--stage',choices=STAGES,required=True); sub.add_argument('--seed',type=int,default=73)
            sub.add_argument('--steps',type=int,default=1024)
            sub.add_argument('--smoke',action='store_true',help='Explicit1or2-update overfit diagnostic; no evaluation/selection')
    return parser


def main():
    args=argument_parser().parse_args()
    if args.command=='cache': result=run_cache(args)
    else:
        values=vars(args); values.pop('command'); result=run_train(TrainConfig(**values))
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__': main()
