"""Matched rank16 readers and immutable official-encoder context-only cache.

The cache phase is a separate process. Training loads the base once, wraps it
in a fresh reader adapter, and never instantiates the rank512 ICAE encoder.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file

from .paired_data import canonical_hash, sha256_file
from .repro_probe import _tensor_bytes

MEMORY_SLOTS=128


def state_identity(state):
    entries={n:dict(shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(_tensor_bytes(t)).hexdigest())
        for n,t in sorted(state.items())}
    return dict(sha256=canonical_hash(entries),tensors=entries,
        parameters=sum(t.numel() for t in state.values()))


def context_key(ids):
    if not isinstance(ids,list) or not ids or any(type(i)!=int or i<0 for i in ids) or len(ids)>512:
        raise ValueError('Require exact nonempty context tokens within 512, without truncation')
    return canonical_hash(ids)


@contextmanager
def encoder_eval(encoder):
    modes=[(module,module.training) for module in encoder.modules()]
    try:
        encoder.eval()
        with torch.no_grad(): yield
    finally:
        for module,mode in modes: module.training=mode


def build_cache(encoder, contexts, output, provenance):
    """Accept context token lists only: no query/answer objects can enter encoder."""
    if encoder.memory_size!=MEMORY_SLOTS:
        raise ValueError('Official cache requires 128 memory vectors')
    unique={context_key(ids):list(ids) for ids in contexts}
    if not unique:
        raise ValueError('Empty cache')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    def status(**values):
        (output/'status.json').write_text(json.dumps(values,indent=2),encoding='utf-8')
        print(json.dumps(values),flush=True)
    status(status='running',phase='cache',completed=0,total=len(unique))
    entries={}
    try:
        with encoder_eval(encoder):
            for index,(key,ids) in enumerate(sorted(unique.items())):
                memory=encoder.compress_ids(torch.tensor([ids],device=encoder.device,dtype=torch.long)).detach().cpu().contiguous()
                if memory.shape!=(1,MEMORY_SLOTS,encoder.hidden_size) or not torch.isfinite(memory).all():
                    raise ValueError('Invalid encoder memory')
                filename=f'{key}.safetensors'
                save_file({'memory':memory},str(output/filename))
                entries[key]=dict(token_ids=ids,file=filename,sha256=sha256_file(output/filename),
                    tensor_sha256=state_identity({'memory':memory})['sha256'])
                status(status='running',phase='cache',completed=index+1,total=len(unique))
            ft=encoder.memory_token_embed.weight[MEMORY_SLOTS+2].detach().cpu().contiguous()
            if not torch.isfinite(ft).all(): raise ValueError('Nonfinite FT delimiter')
            save_file({'ft':ft},str(output/'ft.safetensors'))
        encoder_state={n:p for n,p in encoder.named_parameters() if 'lora_' in n or n.startswith('memory_token_embed.')}
        manifest=dict(schema_version=4,memory_slots=MEMORY_SLOTS,hidden_size=encoder.hidden_size,
            provenance=provenance,contexts=entries,ft_sha256=sha256_file(output/'ft.safetensors'),
            encoder_tensor_identity=state_identity(encoder_state),
            encoder_execution=dict(training=False,gradient_enabled=False,
                base_dtype=str(encoder.base.get_input_embeddings().weight.dtype),
                encoder_dtypes=sorted({str(p.dtype) for p in encoder_state.values()})),
            purpose='Frozen eval/no_grad official ICAE context-only vectors; no query/answer encoder access')
        manifest['manifest_sha256']=canonical_hash(manifest)
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        status(status='success',phase='complete',completed=len(unique),total=len(unique))
        return manifest
    except BaseException as exc:
        status(status='failed',phase='cache',completed=len(entries),error_type=type(exc).__name__,error=str(exc))
        raise


class MemoryCache:
    def __init__(self,path,provenance):
        self.path=Path(path)
        self.manifest=json.loads((self.path/'manifest.json').read_text(encoding='utf-8'))
        contents={k:v for k,v in self.manifest.items() if k!='manifest_sha256'}
        status=json.loads((self.path/'status.json').read_text(encoding='utf-8'))
        if (status['status']!='success' or self.manifest['schema_version']!=4 or self.manifest['memory_slots']!=128
            or self.manifest['provenance']!=provenance or canonical_hash(contents)!=self.manifest['manifest_sha256']):
            raise ValueError('Cache/source/model/provenance mismatch')
        if sha256_file(self.path/'ft.safetensors')!=self.manifest['ft_sha256']:
            raise ValueError('Frozen FT delimiter mismatch')
        self.ft=load_file(str(self.path/'ft.safetensors'))['ft']
        if self.ft.shape!=(self.manifest['hidden_size'],) or not torch.isfinite(self.ft).all():
            raise ValueError('Invalid cached FT tensor')
        for key,entry in self.manifest['contexts'].items():
            if (context_key(entry['token_ids'])!=key or entry['file']!=f'{key}.safetensors'):
                raise ValueError('Malformed context key/path')

    def get(self,ids):
        key=context_key(ids)
        entry=self.manifest['contexts'].get(key)
        if entry is None or ids!=entry['token_ids']:
            raise ValueError('Exact context tokens missing from frozen cache')
        path=self.path/entry['file']
        if sha256_file(path)!=entry['sha256']:
            raise ValueError('Cached tensor file changed')
        memory=load_file(str(path))['memory']
        if (memory.shape!=(1,128,self.manifest['hidden_size']) or not torch.isfinite(memory).all()
            or state_identity({'memory':memory})['sha256']!=entry['tensor_sha256']):
            raise ValueError('Cached tensor shape/content mismatch')
        return memory

    def validate_contexts(self,contexts):
        required={context_key(ids) for ids in contexts}
        if required!=set(self.manifest['contexts']):
            raise ValueError('Cache coverage differs from exact V4 suite contexts')
        for key in sorted(required): self.get(self.manifest['contexts'][key]['token_ids'])


class Reader(nn.Module):
    def __init__(self,base,ft,seed=73):
        super().__init__()
        if hasattr(base,'peft_config'):
            raise ValueError('Reader requires a fresh base without the official encoder adapter')
        if ft.shape!=(base.config.hidden_size,) or not torch.isfinite(ft).all():
            raise ValueError('Frozen FT delimiter shape/content mismatch')
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        self.reader=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
            bias='none',target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
        self.reader.gradient_checkpointing_disable()
        self.register_buffer('ft',ft.detach().clone())
        for p in self.parameters():
            if p.requires_grad: p.data=p.data.float()
        if any('lora_' not in n for n,p in self.named_parameters() if p.requires_grad):
            raise ValueError('Only reader LoRA may train')

    @property
    def base(self): return self.reader.get_base_model()

    @property
    def device(self): return self.base.get_input_embeddings().weight.device

    def embeddings(self,ids):
        return self.base.get_input_embeddings()(torch.as_tensor(ids,device=self.device,dtype=torch.long))

    def prefix(self,item,arm,memory=None):
        context_key(item['context'])
        if arm=='full_reader':
            payload=self.embeddings(item['context'][1:]).unsqueeze(0)
        elif arm=='memory_reader':
            if memory is None or memory.shape!=(1,128,self.base.config.hidden_size) or not torch.isfinite(memory).all():
                raise ValueError('Require exactly 128 finite cached memory vectors')
            payload=memory.detach()
        else: raise ValueError('Unknown reader arm')
        left=self.embeddings(item['left']).unsqueeze(0)
        right=self.embeddings(item['right']).unsqueeze(0)
        return torch.cat((left,payload.to(left),self.ft.reshape(1,1,-1).to(left),right),dim=1)

    def forward_prefix(self,inputs,**kwargs):
        # Never enter disable_adapter(): this is the trainable reader pathway.
        return self.reader(inputs_embeds=inputs,**kwargs)

    def answer_nll(self,prefix,answer):
        if not answer: raise ValueError('Answer including EOS is required')
        if prefix.shape[1]+len(answer)>self.base.config.max_position_embeddings:
            raise ValueError('Sequence exceeds model positional limit; truncation forbidden')
        inputs=torch.cat((prefix,self.embeddings(answer).unsqueeze(0)),dim=1)
        output=self.forward_prefix(inputs,use_cache=False,return_dict=True)
        logits=output.logits[0,prefix.shape[1]-1:prefix.shape[1]-1+len(answer)].float()
        targets=torch.tensor(answer,device=self.device,dtype=torch.long)
        return torch.nn.functional.cross_entropy(logits,targets)

    def adapter_state(self):
        return {n:p.detach().cpu().clone().contiguous() for n,p in self.named_parameters() if p.requires_grad}


def load_reader(base_path,cache,seed=73,device='cuda'):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    base=AutoModelForCausalLM.from_pretrained(base_path,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True,low_cpu_mem_usage=True,device_map={'':device})
    model=Reader(base,cache.ft,seed=seed).to(device)
    tokenizer=AutoTokenizer.from_pretrained(base_path,local_files_only=True,use_fast=False)
    return model,tokenizer
