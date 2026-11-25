"""Official online encoder, matched readers, and bounded exact auxiliary replay."""
from contextlib import contextmanager
import torch
from .v4_training import answer_nll as _answer_nll, prefix_for as _prefix_for
from .v4_model import state_identity

def configure_encoder(model,fixture=False):
    """All existing official q/v LoRA train; memory/FT/base/retrieval never train."""
    from .model import ICAE
    if not isinstance(model,ICAE) or model.memory_size!=128: raise ValueError('Official ICAE interface with128 memory slots required')
    configs=list(model.icae.peft_config.values())
    if len(configs)!=1 or (not fixture and configs[0].r!=512) or set(configs[0].target_modules)!={'q_proj','v_proj'} or configs[0].lora_dropout!=.05:
        raise ValueError('Require every official rank512 q/v encoder LoRA tensor')
    model.requires_grad_(False)
    names=[]
    for name,p in model.named_parameters():
        if '.lora_A.' in name or '.lora_B.' in name:
            if not any('.'+q+'.' in name for q in ('q_proj','v_proj')): raise ValueError('Unexpected LoRA target')
            p.requires_grad_(True); p.data=p.data.float(); names.append(name)
    expected=model.base.config.num_hidden_layers*4
    if len(names)!=expected or model.base.is_gradient_checkpointing: raise ValueError('Incomplete all-layer LoRA or checkpointing enabled')
    return names

def prefix_for(model,item,arm,cache=None):
    # Only context, left and right enter the historical model prefix path.
    if arm=='encoder': arm='encoder_sft'
    return _prefix_for(model,{k:item[k] for k in ('context','left','right')},arm,cache)

def answer_nll(model,prefix,answer,arm):
    return _answer_nll(model,prefix,answer,'encoder_sft' if arm=='encoder' else arm)

def adapter_state(model):
    return {n:p.detach().cpu().clone().contiguous() for n,p in model.named_parameters() if p.requires_grad}

def load_state(model,state):
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(parameters)!=set(state): raise ValueError('Exact trainable tensor names required for reload')
    for name,p in parameters.items():
        if p.shape!=state[name].shape or p.dtype!=state[name].dtype or not torch.isfinite(state[name]).all(): raise ValueError('Reload shape/dtype/finiteness mismatch')
    with torch.no_grad():
        for name,p in parameters.items(): p.copy_(state[name].to(p.device))

@contextmanager
def auxiliary_no_dropout(model):
    modes=[(m,m.training) for m in model.modules()]
    try:
        # Functional attention dropout also consults its module training flag.
        # Eval preserves autograd and disables every stochastic dropout path.
        model.eval()
        yield
    finally:
        for m,mode in modes: m.training=mode

def validate_prepared_pair(pair):
    if len(pair)!=2: raise ValueError('Require two known rows')
    a,b=pair
    if (a['kind']!='swap' or b['kind']!='swap' or a['pair_id']!=b['pair_id'] or {a['variant'],b['variant']}!={0,1}
        or a['right']!=b['right'] or a['left']!=b['left'] or a['context']==b['context']
        or a['answer']!=b['foil'] or b['answer']!=a['foil'] or a['answer']==b['answer']):
        raise ValueError('Prepared question/context/crossed labels violate paired contract')

def pair_objective(scores,method):
    if scores.shape!=(2,2) or method not in ('hn','paired'): raise ValueError('Require2x2 candidate scores and HN/paired')
    if not torch.isfinite(scores).all(): raise FloatingPointError('Nonfinite candidate score')
    s00,s01,s10,s11=scores.reshape(-1)
    hn=.5*(torch.nn.functional.softplus(.5-s00+s01)+torch.nn.functional.softplus(.5-s11+s10))
    if method=='hn': return hn
    column=.5*(torch.nn.functional.softplus(.5-s00+s10)+torch.nn.functional.softplus(.5-s11+s01))
    return .5*(hn+column)

def backward_pair(model,pair,method,scale,return_scores=False):
    """Vector branches replay each context, sharing its two candidate scores.

    Detached score leaves implement the first-order chain rule only. Gradients
    replay through both frozen decoder and encoder, with no optimizer action.
    """
    from .benchmark import _sequential_auxiliary
    validate_prepared_pair(pair)
    pair=sorted(pair,key=lambda r:r['variant'])
    answers=[r['answer'] for r in pair]
    def branch(item):
        # Sequential candidate replay also bounds decoder graph memory. One
        # scalar branch per score yields the identical eight-score objective.
        return [lambda answer=answer: -answer_nll(model,prefix_for(model,item,'encoder'),answer,'encoder') for answer in answers]
    branches=[b for item in pair for b in branch(item)]
    raw=[]
    def objective(*scores):
        matrix=torch.stack(scores).reshape(2,2); raw.extend(matrix.detach().cpu().tolist())
        return pair_objective(matrix,method)
    with auxiliary_no_dropout(model):
        loss=_sequential_auxiliary(branches,objective,model.device,scale)
    return dict(pair_id=pair[0]['pair_id'],row_ids=[r.get('id') for r in pair],answer_token_ids=answers,scores=raw,loss=loss,scale=scale) if return_scores else loss

def load_model(config,cache=None):
    from .model import load_icae
    if config.arm=='encoder':
        model,tokenizer=load_icae(config.base_path,config.adapter_path,device=config.device,dtype=torch.bfloat16)
        configure_encoder(model); return model,tokenizer
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from safetensors.torch import load_file
    from .v4_model import Reader
    ft=(cache.ft if cache is not None else load_file(config.adapter_path,device='cpu')['memory_token_embed.weight'][130])
    base=AutoModelForCausalLM.from_pretrained(config.base_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,low_cpu_mem_usage=True,device_map={'':config.device})
    model=Reader(base,ft,seed=config.seed).to(config.device)
    tokenizer=AutoTokenizer.from_pretrained(config.base_path,local_files_only=True,use_fast=False)
    return model,tokenizer
