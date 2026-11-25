import copy
import importlib
import unittest


def module(test):
    try: value=importlib.import_module('binding.v11_model')
    except ModuleNotFoundError: value=None
    test.assertIsNotNone(value,'Residual relation-preserving memory model missing')
    return value


class ProtocolTests(unittest.TestCase):
    def test_closed_supervision_choices(self):
        m=module(self)
        self.assertEqual(m.objective_weights('sft'),(0.,0.))
        self.assertEqual(m.objective_weights('relation'),(.10,.05))
        with self.assertRaises(ValueError):m.objective_weights('best')


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Neural tests run on the remote installed Torch/PEFT runtime')
class NeuralTests(unittest.TestCase):
    def make(self):
        from transformers import MistralConfig,MistralForCausalLM
        from binding.model import ICAE
        torch.set_num_threads(1);torch.manual_seed(11)
        config=MistralConfig(vocab_size=64,hidden_size=16,intermediate_size=24,num_hidden_layers=1,
                             num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=1024,pad_token_id=0)
        config._attn_implementation='sdpa'
        model=ICAE(MistralForCausalLM(config),memory_size=128,rank=2,projection_dim=4)
        with torch.no_grad():
            for name,p in model.named_parameters():
                if 'lora_B.default' in name:p.normal_(0,.02)
        return model

    def unit(self):
        return dict(contexts=[[1,3,4],[1,3,5],[1,3,6],[1,3,7]],
                    queries=[dict(left=[1,5],right=[6,7]),dict(left=[1,5],right=[6,8]),dict(left=[1,5],right=[6,9])],
                    candidates=[[10,2],[11,2],[12,2]],unaffected_answer=[13,2])

    def test_residual_initialization_preserves_memory_and_freezes_original(self):
        m=module(self);model=self.make();model.eval()
        before=model.compress_ids(torch.tensor([[1,3,4]])).detach()
        old={n:p.detach().clone() for n,p in model.named_parameters()}
        names=m.configure_residual(model,seed=73,rank=2,fixture=True)
        self.assertTrue(names)
        self.assertTrue(all('.relation.' in name for name in names))
        self.assertTrue(all(p.requires_grad==('.relation.' in n) for n,p in model.named_parameters()))
        self.assertTrue(all(not part.training for part in model.modules()))
        self.assertTrue(torch.equal(before,model.compress_ids(torch.tensor([[1,3,4]])).detach()))
        self.assertTrue(all(torch.equal(value,dict(model.named_parameters())[name]) for name,value in old.items()))

    def test_decoder_context_restores_exact_trainability_after_peft_reenable(self):
        m=module(self);model=self.make();m.configure_residual(model,73,rank=2,fixture=True)
        flags={n:p.requires_grad for n,p in model.named_parameters()}
        memory=model.compress_ids(torch.tensor([[1,3,4]]))
        logits=m.decoder_logits(model,memory,dict(left=[1,5],right=[6,7]),[10,2])
        logits.sum().backward()
        self.assertEqual(flags,{n:p.requires_grad for n,p in model.named_parameters()})
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for n,p in model.named_parameters() if '.relation.' in n))
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if '.relation.' not in n))

    def test_memory_relay_matches_direct_full_objective_gradient(self):
        m=module(self)
        for method in ('sft','relation'):
            direct=self.make();m.configure_residual(direct,73,rank=2,fixture=True)
            relay=copy.deepcopy(direct);unit=self.unit()
            memories=[direct.compress_ids(torch.tensor([context])) for context in unit['contexts']]
            ce=torch.nn.functional.cross_entropy;sp=torch.nn.functional.softplus
            sft=[];scores=[];unaffected=[];gold=(0,1,2,0)
            for index,memory in enumerate(memories):
                candidates=[]
                for answer in unit['candidates']:
                    logits=m.decoder_logits(direct,memory,unit['queries'][0],answer)
                    candidates.append(-ce(logits,torch.tensor(answer)))
                scores.append(torch.stack(candidates));sft.append(-candidates[gold[index]])
                logits=m.decoder_logits(direct,memory,unit['queries'][1],unit['unaffected_answer'])
                unaffected.append(logits);sft.append(ce(logits,torch.tensor(unit['unaffected_answer'])))
                logits=m.decoder_logits(direct,memory,unit['queries'][2],unit['candidates'][2])
                sft.append(ce(logits,torch.tensor(unit['candidates'][2])))
            loss=torch.stack(sft).mean()
            a,b=m.objective_weights(method)
            margins=torch.stack([sp(.5-scores[i][gold[i]]+scores[i][j]) for i in range(4) for j in range(3) if j!=gold[i]]).mean()
            teacher=unaffected[0].detach().softmax(-1)
            locality=torch.stack([torch.nn.functional.kl_div(value.log_softmax(-1),teacher,reduction='batchmean') for value in unaffected[1:]]).mean()
            loss=loss+a*margins+b*locality
            loss.backward()
            result=m.backward_unit(relay,unit,method)
            self.assertAlmostEqual(result['loss'],float(loss.detach()),places=5)
            for (name,p),(_,q) in zip(direct.named_parameters(),relay.named_parameters()):
                if p.requires_grad:
                    self.assertIsNotNone(q.grad,name)
                    self.assertTrue(torch.allclose(p.grad,q.grad,atol=3e-6,rtol=5e-4),name)
                else:self.assertIsNone(q.grad,name)


if __name__=='__main__':unittest.main(verbosity=2)
