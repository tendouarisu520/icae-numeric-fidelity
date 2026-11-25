"""Fixed-budget ICAE V2 adapter-compatible model with one frozen base.

Reference: getao/icae commit 469a46886a92dd5e76b2d12a8bac0fb7ed7d4cdd.
The decoder shares base weights, has adapters disabled, and retains input
gradients. Checkpointing is deliberately disabled: recomputation must never
observe the wrong adapter state. This is verified against a separate decoder.
"""
from contextlib import nullcontext

import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from .objectives import sequence_logp


class ICAE(nn.Module):
    def __init__(self, base, memory_size=128, rank=512, dropout=.05, projection_dim=256):
        super().__init__()
        self.memory_size = memory_size
        self.hidden_size = base.config.hidden_size
        self.icae = get_peft_model(base, LoraConfig(
            r=rank, lora_alpha=32, lora_dropout=dropout, bias='none',
            target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        self.icae.gradient_checkpointing_disable()
        self.memory_token_embed = nn.Embedding(memory_size + 3, self.hidden_size)
        self.retrieval_proj = nn.Linear(self.hidden_size, projection_dim, bias=False)

    @property
    def base(self):
        return self.icae.get_base_model()

    @property
    def device(self):
        return self.base.get_input_embeddings().weight.device

    def embeddings(self, ids):
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        return self.base.get_input_embeddings()(ids)

    def compress_ids(self, context_ids):
        if context_ids.ndim != 2 or context_ids.shape[0] != 1:
            raise ValueError('Pilot compressor supports microbatch 1 without padding')
        if self.base.is_gradient_checkpointing:
            raise RuntimeError('Shared adapter/base requires checkpointing disabled')
        embeddings = self.embeddings(context_ids)
        memory = self.memory_token_embed.weight[:self.memory_size].unsqueeze(0).to(embeddings)
        inputs = torch.cat((embeddings, memory), dim=1)
        hidden = self.base.model(inputs_embeds=inputs, use_cache=False, return_dict=True).last_hidden_state
        return hidden[:, -self.memory_size:]

    def prefix_ids(self, memory, left, right):
        # right is the query and closing [/INST]; the learned FT token is inserted.
        le = self.embeddings(left).unsqueeze(0)
        re = self.embeddings(right).unsqueeze(0)
        ft = self.memory_token_embed.weight[self.memory_size + 2].reshape(1, 1, -1).to(le)
        return torch.cat((le, memory.to(le), ft, re), dim=1)

    def score_ids(self, memory, left, right, answers, decoder=None):
        prefix = self.prefix_ids(memory, left, right)
        length = prefix.shape[1]
        max_answer = max(len(a) for a in answers)
        parts, masks, targets = [], [], []
        for answer in answers:
            if not answer:
                raise ValueError('Empty answer')
            pad = max_answer - len(answer)
            parts.append(torch.cat((prefix, self.embeddings(answer + [0] * pad).unsqueeze(0)), dim=1))
            masks.append([1] * (length + len(answer)) + [0] * pad)
            targets.append([-100] * length + answer + [-100] * pad)
        inputs = torch.cat(parts, dim=0)
        with self.icae.disable_adapter() if decoder is None else nullcontext():
            out = (self.base if decoder is None else decoder)(
                inputs_embeds=inputs,
                attention_mask=torch.tensor(masks, device=self.device),
                use_cache=False, return_dict=True)
        return sequence_logp(out.logits, torch.tensor(targets, device=self.device))

    def query_vector(self, ids):
        with torch.no_grad(), self.icae.disable_adapter():
            hidden = self.base.model(input_ids=ids, use_cache=False).last_hidden_state.mean(dim=1)
        return self.retrieval_proj(hidden.to(self.retrieval_proj.weight.dtype))

    def memory_vector(self, memory):
        return self.retrieval_proj(memory.mean(dim=1).to(self.retrieval_proj.weight.dtype))

    @torch.no_grad()
    def generate_ids(self, memory, left, right, eos_id, max_new_tokens=16):
        inputs = self.prefix_ids(memory, left, right)
        generated, cache = [], None
        with self.icae.disable_adapter():
            for _ in range(max_new_tokens):
                out = self.base(inputs_embeds=inputs, past_key_values=cache, use_cache=True)
                next_id = int(out.logits[0, -1].argmax())
                if next_id == eos_id:
                    break
                generated.append(next_id)
                cache = out.past_key_values
                inputs = self.embeddings([next_id]).unsqueeze(0)
        return generated

    def adapter_state(self, include_projection=True):
        return {n: p.detach().cpu().contiguous() for n, p in self.named_parameters()
                if p.requires_grad and (include_projection or not n.startswith('retrieval_proj.'))}

    def load_adapter_state(self, state):
        current = dict(self.named_parameters())
        required = {n for n, p in current.items() if p.requires_grad and not n.startswith('retrieval_proj.')}
        allowed = {n for n, p in current.items() if p.requires_grad}
        if not required <= state.keys() or not state.keys() <= allowed:
            raise ValueError(f'Adapter mismatch: missing={sorted(required - state.keys())[:4]}, unexpected={sorted(state.keys() - allowed)[:4]}')
        for name, tensor in state.items():
            if current[name].shape != tensor.shape:
                raise ValueError(f'Shape mismatch: {name}')
        with torch.no_grad():
            for name, tensor in state.items():
                current[name].copy_(tensor.to(current[name]))


def load_icae(base_path, adapter_path, device='cuda', dtype=torch.bfloat16, memory_size=128):
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # local_files_only prevents an accidental untracked large download at run time.
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype,
                                               attn_implementation='sdpa', local_files_only=True,
                                               low_cpu_mem_usage=True, device_map={'': device})
    model = ICAE(base, memory_size=memory_size)
    model.load_adapter_state(load_file(adapter_path))
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True, use_fast=False)
    return model, tokenizer

