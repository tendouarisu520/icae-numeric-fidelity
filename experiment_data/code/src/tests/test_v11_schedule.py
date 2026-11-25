import importlib
import unittest


class ScheduleTests(unittest.TestCase):
    def test_same_seed_has_same_complete_unit_exposure(self):
        m=importlib.import_module('binding.v11_training')
        ids=[str(i) for i in range(8)]
        a=m.schedule(ids,73,32);b=m.schedule(ids,73,32)
        self.assertEqual(a,b)
        for start in range(0,32,8):self.assertEqual(set(a[start:start+8]),set(ids))
        self.assertNotEqual(a,m.schedule(ids,97,32))
        with self.assertRaises(ValueError):m.schedule(ids,73,1025)

    def test_overfit_gate_requires_both_answering_and_selectivity(self):
        m=importlib.import_module('binding.v11_training')
        rates=dict(target_all_four=1.,known_exact=1.,unknown_exact=1.)
        self.assertTrue(m.overfit_pass(dict(units=8,rates=rates)))
        rates['known_exact']=.5
        self.assertFalse(m.overfit_pass(dict(units=8,rates=rates)))
        rates['known_exact']=1.;rates['target_all_four']=.75
        self.assertFalse(m.overfit_pass(dict(units=8,rates=rates)))


if __name__=='__main__':unittest.main(verbosity=2)
