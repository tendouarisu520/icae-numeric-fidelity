import importlib
import copy
import unittest

def module(name):
    try:return importlib.import_module('binding.'+name)
    except ModuleNotFoundError:return None

class ProtocolTests(unittest.TestCase):
    @staticmethod
    def comparison():
        return dict(status='complete',endpoint=512,cells=[dict(arm=a,method=m,seed=s) for s in (73,97)
            for a,m in (('compressed','sft'),('compressed','assignment'),('compressed','relation'),('full','sft'))],
            comparisons={m:{str(s):dict(continuation_gate=False) for s in (73,97)} for m in ('assignment','relation')})

    def test_followup_requires_two_seed_failure_of_both_original_methods(self):
        m=module('v14_protocol');self.assertTrue(hasattr(m,'should_run'),'Conditional budget gate missing')
        comparison=self.comparison();before=copy.deepcopy(comparison)
        self.assertTrue(m.should_run(comparison));self.assertEqual(comparison,before)
        comparison['comparisons']['assignment']['73']['continuation_gate']=True
        self.assertTrue(m.should_run(comparison))
        comparison['comparisons']['assignment']['97']['continuation_gate']=True
        self.assertFalse(m.should_run(comparison))

    def test_incomplete_original_study_cannot_trigger_followup(self):
        m=module('v14_protocol');self.assertTrue(hasattr(m,'should_run'),'Conditional budget gate missing')
        for field,value in (('status','running'),('endpoint',256),('cells',self.comparison()['cells'][:-1])):
            comparison=self.comparison();comparison[field]=value
            with self.assertRaises(ValueError):m.should_run(comparison)

    def test_soft_min_bounds_every_conditional_gap(self):
        m=module('v14_protocol');self.assertIsNotNone(m,'Bottleneck protocol missing')
        for values in ([5.,-1.,4.,0.],[1.,1.,1.,1.],[-4.,-3.,2.,1.]):self.assertLessEqual(m.soft_min(values),min(values))
        with self.assertRaises(ValueError):m.soft_min([1.,float('nan')])

    def test_correct_pair_preferred_to_compensated_wrong_condition(self):
        m=module('v14_protocol');self.assertIsNotNone(m,'Bottleneck protocol missing')
        safe=[[-4.,-5.],[-5.,-4.]];unsafe=[[-1.,-6.],[-5.,-6.]]
        self.assertLess(m.penalty(safe),m.penalty(unsafe))
        self.assertEqual(m.gaps(unsafe),[5.,-1.,4.,0.])

    def test_new_objective_available_without_mutating_original(self):
        self.assertIsNotNone(module('v14_objectives'),'Bottleneck objective missing')

try:import torch
except ImportError:torch=None

@unittest.skipIf(torch is None,'Actual neural verification runs remotely')
class NeuralTests(unittest.TestCase):
    def test_torch_score_and_worst_gap_gradient(self):
        from binding.v14_objectives import assignment
        from binding.v14_protocol import penalty
        scores=torch.tensor([[-1.,-6.],[-5.,-6.]],requires_grad=True)
        loss=assignment(scores);self.assertAlmostEqual(float(loss),penalty(scores.detach().tolist()),places=6)
        loss.backward();self.assertTrue(bool(torch.isfinite(scores.grad).all()))
        self.assertLess(float(scores.grad[1,1]),0)

    def test_actual_reader_gradient_and_memory_immutability(self):
        from test_v13_neural import NeuralTests as Fixture
        from binding.v14_objectives import backward
        model,item,cache=Fixture.fixture(self);before=cache.digest()
        result=backward(model,item,cache,'compressed','bottleneck',1.)
        self.assertGreater(result['bottleneck_loss'],0)
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        self.assertEqual(before,cache.digest())

if __name__=='__main__':unittest.main()
