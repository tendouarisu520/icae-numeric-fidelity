import copy
import importlib
import unittest


def module(test):
    try:m=importlib.import_module('binding.v12_reader')
    except ModuleNotFoundError:m=None
    test.assertIsNotNone(m,'Frozen-memory reader implementation missing')
    return m


class EntryTests(unittest.TestCase):
    def test_implementation_available(self):module(self)


try:import torch
except ImportError:torch=None


@unittest.skipIf(torch is None,'Actual neural tests run in installed remote Torch environment')
class NeuralTests(unittest.TestCase):
    def fixture(self):
        from transformers import MistralConfig,MistralForCausalLM
        from binding.model import ICAE
        from binding.v4_model import Reader
        torch.set_num_threads(1);torch.manual_seed(11)
        cfg=MistralConfig(vocab_size=64,hidden_size=16,intermediate_size=24,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=1024,pad_token_id=0)
        cfg._attn_implementation='sdpa';base=MistralForCausalLM(cfg)
        encoder=ICAE(copy.deepcopy(base),rank=2,projection_dim=4).eval().requires_grad_(False)
        reader=Reader(base,encoder.memory_token_embed.weight[130],seed=73).eval()
        class Tokenizer:
            eos_token_id=2
            def decode(self,ids,skip_special_tokens=True):return ' '.join(map(str,ids))
        return encoder,reader,Tokenizer()

    def test_context_cache_is_frozen_exact_and_deduplicated(self):
        m=module(self);encoder,reader,_=self.fixture();contexts=[[1,3,4],[1,3,4],[1,3,5]]
        cache=m.build_ram_cache(encoder,contexts)
        self.assertEqual(len(cache.entries),2)
        with torch.no_grad():expected=encoder.compress_ids(torch.tensor([contexts[0]]))
        self.assertTrue(torch.equal(cache.get(contexts[0]),expected))
        self.assertFalse(cache.get(contexts[0]).requires_grad)
        with self.assertRaises(ValueError):cache.get([1,3,6])
        with self.assertRaises(ValueError):m.build_ram_cache(encoder,[{'context':[1,3]}])
        encoder.memory_token_embed.weight.requires_grad_(True)
        with self.assertRaises(ValueError):m.build_ram_cache(encoder,contexts)

    def test_zero_reader_reproduces_frozen_decoder_and_prefix(self):
        m=module(self);encoder,reader,tok=self.fixture();ids=[1,3,4];q=dict(left=[1,5],right=[6,7])
        cache=m.build_ram_cache(encoder,[ids]);mem=cache.get(ids)
        a=encoder.prefix_ids(mem,q['left'],q['right']);b=reader.prefix(dict(context=ids,**q),'memory_reader',mem)
        self.assertTrue(torch.equal(a,b))
        from binding.v11_training import generate
        self.assertEqual(generate(encoder,a,tok,'encoder'),m.greedy(reader,b,tok))

    def test_scores_align_with_supervised_loss_and_cache_does_not_change(self):
        m=module(self);encoder,reader,_=self.fixture();ids=[1,3,4]
        cache=m.build_ram_cache(encoder,[ids]);before=cache.digest()
        prefix=reader.prefix(dict(context=ids,left=[1,5],right=[6,7]),'memory_reader',cache.get(ids))
        score=m.score_ids(reader,prefix,[10,2]);loss=reader.answer_nll(prefix,[10,2])
        self.assertAlmostEqual(score['mean'],-float(loss.detach()),places=6)
        self.assertAlmostEqual(score['sum'],score['mean']*2,places=6)
        self.assertEqual(score['tokens'],2)
        loss.backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.count_nonzero()) for p in reader.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for p in reader.parameters() if not p.requires_grad))
        self.assertEqual(before,cache.digest())


if __name__=='__main__':unittest.main()
