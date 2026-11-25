import importlib
import unittest


class EntryTests(unittest.TestCase):
    def test_full_generation_entry_exists(self):
        try:
            value=importlib.import_module('binding.v11_full_entry')
        except ModuleNotFoundError:
            value=None
        self.assertIsNotNone(value,'Full reader generation compatibility entry is missing')


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Run actual neural tests on remote Torch runtime')
class GenerationTests(unittest.TestCase):
    def make(self):
        from transformers import MistralConfig,MistralForCausalLM
        from binding.v4_model import Reader
        torch.set_num_threads(1)
        config=MistralConfig(vocab_size=64,hidden_size=16,intermediate_size=24,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
            max_position_embeddings=1024,pad_token_id=0)
        config._attn_implementation='sdpa'
        model=Reader(MistralForCausalLM(config),torch.zeros(16),seed=73).eval()
        class Tokenizer:
            eos_token_id=2
            def decode(self,ids,skip_special_tokens=True):return ' '.join(map(str,ids))
        return model,Tokenizer()

    def test_original_full_generation_reproduces_keyword_failure(self):
        from binding.v11_training import generate
        model,tokenizer=self.make()
        with self.assertRaisesRegex(TypeError,'inputs'):
            generate(model,model.embeddings([1,3,4]).unsqueeze(0),tokenizer,'full')

    def test_fixed_generation_matches_direct_positional_reader(self):
        from binding.v11_full_entry import generate
        model,tokenizer=self.make()
        prefix=model.embeddings([1,3,4]).unsqueeze(0)
        expected=[];cache=None;inputs=prefix
        with torch.no_grad():
            for _ in range(32):
                out=model.forward_prefix(inputs,past_key_values=cache,use_cache=True,return_dict=True)
                token=int(out.logits[0,-1].argmax());expected.append(token)
                if token==2:break
                cache=out.past_key_values;inputs=model.embeddings([token]).unsqueeze(0)
        actual=generate(model,prefix,tokenizer,'full')
        self.assertEqual(actual,(tokenizer.decode(expected),dict(token_ids=expected,eos_emitted=expected[-1]==2)))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_reader_training_path_still_has_only_adapter_gradients(self):
        model,_=self.make()
        loss=model.answer_nll(model.embeddings([1,3,4]).unsqueeze(0),[5,2]);loss.backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.count_nonzero()) for p in model.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))


if __name__=='__main__':unittest.main()
