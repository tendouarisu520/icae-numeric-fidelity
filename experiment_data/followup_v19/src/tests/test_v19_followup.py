import unittest
from collections import Counter
from binding.v13_data import make_probe
from binding.v19_data import transform,balanced_training,prospective_units
from binding.v19_analysis import loss_components,primary_contrasts

class FollowupTests(unittest.TestCase):
 def test_no_standard_library_shadow(self):
  from pathlib import Path
  from importlib.machinery import PathFinder
  folder=Path(__file__).resolve().parents[2]/'temp/v19'
  self.assertIsNone(PathFinder.find_spec('queue',[str(folder)]),'Runner must not shadow Python queue')
 def source(self,i):return dict(id=f'source-{i}',article=f'article-{i}',contexts=['Neutral background.'])
 def test_all_levels_and_gold(self):
  for n in (2,3,4):
   u=make_probe(self.source(1),'train',n)
   for s in range(3):
    for p in range(3):
     for e in range(2):
      v=transform(u,s,p,e)
      self.assertEqual(u['answers'],v['answers'])
      for facts,answers in zip(v['facts'],v['answers']):
       mapping={tuple(f['key']):f['value'] for f in facts}
       self.assertEqual([mapping[tuple(k)] for k in v['query_keys']],answers)
       self.assertEqual(6*n,len(mapping))
   self.assertEqual(u['contexts'],transform(u,0,0,0)['contexts'])
 def test_source_and_factor_exposure(self):
  rows=[make_probe(self.source(i),'train',2+i%3) for i in range(512)]
  out=balanced_training(rows)
  self.assertEqual(1024,len(out));self.assertEqual({2},set(Counter(u['source_id'] for u in out).values()))
  cells=Counter((u['semantic_rotation'],u['layout_rotation'],u['target_team_position']) for u in out)
  self.assertEqual(18,len(cells));self.assertLessEqual(max(cells.values())-min(cells.values()),1)
  self.assertEqual(out,balanced_training(rows))
 def test_fresh_cohort(self):
  ordinary,crossed=prospective_units()
  self.assertEqual((96,144),(len(ordinary),len(crossed)))
  self.assertEqual(32,len({u['source_id'] for u in ordinary}))
  self.assertTrue(all(u['source_id'].startswith('v19-prospective-20261108-') for u in ordinary))
  self.assertEqual((ordinary,crossed),prospective_units())
 def test_loss_accounting(self):
  row=dict(loss=2.625,families=[dict(loss=1.05,sft_loss=2,assignment_loss=0,locality_loss=2),dict(loss=1.575,sft_loss=3,assignment_loss=.5,locality_loss=1)])
  x=loss_components(row);self.assertAlmostEqual(x['weighted_ce'],2.5);self.assertAlmostEqual(x['weighted_aux'],.05);self.assertAlmostEqual(x['weighted_locality'],.075)
  row['loss']=7
  with self.assertRaises(ValueError):loss_components(row)
 def test_paired_scene_bootstrap_and_absent_denominator(self):
  rows=[dict(source_id=str(i),target_n=8,choice_correct=8,greedy_correct=4,absent_n=2,digit_advantage_sum=3.) for i in range(4)]
  out=primary_contrasts(rows,resamples=100)
  self.assertEqual(out['choice_generation_gap']['estimate'],.5)
  self.assertEqual(out['absent_digit_advantage']['estimate'],1.5)
  self.assertTrue(out['both_positive'])
  for r in rows:r.update(absent_n=0,digit_advantage_sum=0.)
  self.assertIsNone(primary_contrasts(rows,resamples=100)['absent_digit_advantage']['estimate'])
if __name__=='__main__':unittest.main()
