import copy
import importlib
import unittest

try:import torch
except ImportError:torch=None

class Availability(unittest.TestCase):
    def test_objectives_available(self):
        try:m=importlib.import_module('binding.v13_objectives')
        except ModuleNotFoundError:m=None
        self.assertIsNotNone(m,'V13 objective implementation missing')

@unittest.skipIf(torch is None,'Neural verification requires the installed remote Torch runtime')
class NeuralTests(unittest.TestCase):
    def fixture(self):
        from transformers import MistralConfig,MistralForCausalLM
        from binding.model import ICAE
        from binding.v4_model import Reader
        from binding.v12_reader import build_ram_cache
        torch.set_num_threads(1);torch.manual_seed(13)
        cfg=MistralConfig(vocab_size=64,hidden_size=16,intermediate_size=24,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=1024,pad_token_id=0)
        cfg._attn_implementation='sdpa';base=MistralForCausalLM(cfg)
        enc=ICAE(copy.deepcopy(base),rank=2,projection_dim=4).eval().requires_grad_(False)
        model=Reader(base,enc.memory_token_embed.weight[130],seed=73).eval()
        item=dict(id='fixture',family='probe',contexts=[[1,3,4],[1,3,5],[1,3,6],[1,3,7]],
            queries=[dict(left=[1,8],right=[9+j]) for j in range(4)],
            answers=[[[10,2],[11,2],[12,2],[13,2]],[[11,2],[10,2],[12,2],[13,2]],
                     [[10,2],[11,2],[12,2],[13,2]],[[10,2],[11,2],[12,2],[13,2]]])
        return model,item,build_ram_cache(enc,item['contexts'])

    def test_assignment_has_correct_orientation_and_gradients(self):
        from binding.v13_objectives import assignment
        good=torch.tensor([[2.,-2.],[-2.,2.]],requires_grad=True)
        bad=good.detach().flip(1).requires_grad_(True)
        self.assertLess(float(assignment(good)),float(assignment(bad)))
        assignment(good).backward()
        self.assertLess(float(good.grad[0,0]),0);self.assertGreater(float(good.grad[0,1]),0)

    def test_locality_teacher_detached_and_identical_is_zero(self):
        from binding.v13_objectives import locality
        teacher=torch.randn(3,64,requires_grad=True);student=teacher.detach().clone().requires_grad_(True)
        self.assertAlmostEqual(float(locality(student,teacher)),0.,places=6)
        locality(student+.1*torch.randn_like(student),teacher).backward()
        self.assertIsNone(teacher.grad);self.assertIsNotNone(student.grad)

    def test_sequential_sft_matches_direct_gradients(self):
        from binding.v13_objectives import backward,prefix
        model,item,cache=self.fixture();before=cache.digest()
        reference=sum(model.answer_nll(prefix(model,item,i,j,cache,'compressed'),a)/16
            for i,row in enumerate(item['answers']) for j,a in enumerate(row))
        reference.backward();expected={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        model.zero_grad(set_to_none=True);result=backward(model,item,cache,'compressed','sft',1.)
        self.assertAlmostEqual(result['loss'],float(reference.detach()),places=5)
        for n,p in model.named_parameters():
            if p.requires_grad:self.assertTrue(torch.allclose(p.grad,expected[n],rtol=3e-4,atol=2e-6),n)
            else:self.assertIsNone(p.grad)
        self.assertEqual(before,cache.digest())

    def test_relation_gradients_are_finite_and_cache_unchanged(self):
        from binding.v13_objectives import backward
        model,item,cache=self.fixture();before=cache.digest()
        result=backward(model,item,cache,'compressed','relation',1.)
        self.assertGreater(result['assignment_loss'],0)
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        self.assertEqual(before,cache.digest())

if __name__=='__main__':unittest.main()
