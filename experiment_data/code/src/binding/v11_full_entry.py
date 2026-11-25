"""Full-reader entry: preserve V11 training, correct positional forward API.

Original pinned sources are not edited. This entry only replaces greedy
generation for the full arm, before calling the original training main.
"""
from . import v11_training as driver

_encoder_generate=driver.generate


def generate(model,inputs,tokenizer,arm):
    if arm=='encoder':return _encoder_generate(model,inputs,tokenizer,arm)
    if arm!='full':raise ValueError('Registered V11 arm required')
    import torch
    generated=[];cache=None
    with torch.no_grad():
        for _ in range(32):
            output=model.forward_prefix(inputs,past_key_values=cache,use_cache=True,return_dict=True)
            token=int(output.logits[0,-1].argmax());generated.append(token)
            if token==tokenizer.eos_token_id:break
            cache=output.past_key_values;inputs=model.embeddings([token]).unsqueeze(0)
    return tokenizer.decode(generated,skip_special_tokens=True),dict(token_ids=generated,eos_emitted=generated[-1]==tokenizer.eos_token_id)


def main():
    driver.generate=generate
    driver.main()


if __name__=='__main__':main()
