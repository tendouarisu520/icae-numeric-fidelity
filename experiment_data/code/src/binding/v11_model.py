"""Small residual memory writer with relation/locality training and gradient relay."""
from contextlib import contextmanager

GOLD=(0,1,2,0)


def objective_weights(method):
    if method=='sft':return 0.,0.
    if method=='relation':return .10,.05
    raise ValueError('Only matched SFT and relation supervision are supported')


def configure_residual(model,seed=73,rank=16,fixture=False):
    import torch
    from peft import LoraConfig
    from .model import ICAE
    if not isinstance(model,ICAE) or model.memory_size!=128 or set(model.icae.peft_config)!={'default'}:
        raise ValueError('Unmodified official ICAE initialization required')
    original=model.icae.peft_config['default']
    if not fixture and (original.r!=512 or rank!=16):
        raise ValueError('Production uses frozen rank512 plus trainable rank16')
    if set(original.target_modules)!={'q_proj','v_proj'} or seed not in (73,97):
        raise ValueError('Fixed q/v targets and registered seeds required')
    model.requires_grad_(False)
    torch.manual_seed(seed)
    model.icae.add_adapter('relation',LoraConfig(r=rank,lora_alpha=32,lora_dropout=0.,bias='none',
                                               target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.icae.base_model.set_adapter(['default','relation'])
    # PEFT activation changes requires_grad. Freeze explicitly after activation.
    model.requires_grad_(False)
    names=[]
    for name,parameter in model.named_parameters():
        if '.lora_A.relation.' in name or '.lora_B.relation.' in name:
            parameter.data=parameter.data.float();parameter.requires_grad_(True);names.append(name)
    if len(names)!=model.base.config.num_hidden_layers*4 or model.base.is_gradient_checkpointing:
        raise ValueError('Incomplete residual or unsupported checkpointing')
    if not fixture and sum(p.numel() for p in model.parameters() if p.requires_grad)!=6815744:
        raise ValueError('Expected6,815,744 residual parameters')
    model.eval()
    return names


@contextmanager
def frozen_decoder(model):
    # PEFT0.13 re-enables gradients on ALL active adapters when leaving its
    # disable_adapter context. Preserve flags so the original writer stays frozen.
    flags=[(parameter,parameter.requires_grad) for parameter in model.parameters()]
    try:
        with model.icae.disable_adapter():
            yield
    finally:
        for parameter,flag in flags:
            parameter.requires_grad_(flag)


def decoder_logits(model,memory,query,answer):
    import torch
    if not answer:
        raise ValueError('Answer+EOS tokens required')
    prefix=model.prefix_ids(memory,query['left'],query['right'])
    if prefix.shape[1]+len(answer)>model.base.config.max_position_embeddings:
        raise ValueError('Decoder position budget exceeded')
    inputs=torch.cat((prefix,model.embeddings(answer).unsqueeze(0)),dim=1)
    with frozen_decoder(model):
        output=model.base(inputs_embeds=inputs,use_cache=False,return_dict=True)
    return output.logits[0,prefix.shape[1]-1:prefix.shape[1]-1+len(answer)].float()


def nll(model,memory,query,answer):
    import torch
    return torch.nn.functional.cross_entropy(decoder_logits(model,memory,query,answer),
                                              torch.tensor(answer,device=model.device))


def backward_unit(model,unit,method):
    import torch
    from .benchmark import _sequential_auxiliary
    a,b=objective_weights(method)
    contexts,queries,candidates=unit['contexts'],unit['queries'],unit['candidates']
    if len(contexts)!=4 or len(queries)!=3 or len(candidates)!=3 or len({tuple(x) for x in candidates})!=3:
        raise ValueError('Four contexts, three questions and three distinct target candidates required')
    if any(part.training for part in model.modules()):
        raise ValueError('Deterministic no-dropout training/replay required')
    flags={name:parameter.requires_grad for name,parameter in model.named_parameters()}
    with torch.no_grad():
        memories=[model.compress_ids(torch.tensor([context],device=model.device)) for context in contexts]
    leaves=[memory.detach().requires_grad_(True) for memory in memories]
    sft_target=0.;margin_value=0.;scores=[]
    if method=='sft':
        for i,memory in enumerate(leaves):
            loss=nll(model,memory,queries[0],candidates[GOLD[i]])/12
            sft_target+=float(loss.detach());loss.backward()
    else:
        branches=[lambda i=i,answer=answer: -nll(model,leaves[i],queries[0],answer)
                  for i in range(4) for answer in candidates]
        captured={}
        def objective(*values):
            matrix=torch.stack(values).reshape(4,3)
            task=-sum(matrix[i,GOLD[i]] for i in range(4))/12
            margin=torch.stack([torch.nn.functional.softplus(.5-matrix[i,GOLD[i]]+matrix[i,j])
                                for i in range(4) for j in range(3) if j!=GOLD[i]]).mean()
            captured.update(sft=float(task.detach()),margin=float(margin.detach()),scores=matrix.detach().cpu().tolist())
            return task+a*margin
        _sequential_auxiliary(branches,objective,model.device,1.)
        sft_target,margin_value,scores=captured['sft'],captured['margin'],captured['scores']
    sft_other=0.;locality_value=0.;teacher=None
    for i,memory in enumerate(leaves):
        logits=decoder_logits(model,memory,queries[1],unit['unaffected_answer'])
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor(unit['unaffected_answer'],device=model.device))/12
        sft_other+=float(loss.detach())
        if i==0:
            teacher=logits.detach().softmax(-1)
        elif b:
            locality=torch.nn.functional.kl_div(logits.log_softmax(-1),teacher,reduction='batchmean')/3
            locality_value+=float(locality.detach());loss=loss+b*locality
        loss.backward()
        del logits,loss
        loss=nll(model,memory,queries[2],candidates[2])/12
        sft_other+=float(loss.detach());loss.backward()
    memory_gradient_norms=[]
    for context,leaf in zip(contexts,leaves):
        if leaf.grad is None or not bool(torch.isfinite(leaf.grad).all()):
            raise FloatingPointError('Missing/nonfinite accumulated memory gradient')
        memory_gradient_norms.append(float(leaf.grad.float().norm()))
        memory=model.compress_ids(torch.tensor([context],device=model.device))
        memory.backward(leaf.grad)
        del memory
    if flags!={name:parameter.requires_grad for name,parameter in model.named_parameters()}:
        raise RuntimeError('Adapter trainability changed during decoder/replay')
    sft=sft_target+sft_other
    return dict(sft_loss=sft,margin_loss=margin_value,locality_loss=locality_value,
                loss=sft+a*margin_value+b*locality_value,memory_gradient_norms=memory_gradient_norms,
                candidate_scores=scores,writing_contexts=4,supervised_answers=12,
                encoder_forward_count=8,query_independent_writing=True,decoder_frozen=True)
