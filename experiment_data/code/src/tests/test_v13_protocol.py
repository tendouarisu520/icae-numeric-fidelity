import importlib
import unittest

def implementation(name):
    try:return importlib.import_module('binding.'+name)
    except ModuleNotFoundError:return None

def source(i='sample',article='article'):
    return dict(id=i,article=article,contexts=['A neutral background paragraph.']*4,
        queries=['Target?','Unaffected?','Unknown?'],target_answers=['old','new','UNKNOWN','old'],
        candidates=['old','new','UNKNOWN'],unaffected_answer='kept')

class ProtocolTests(unittest.TestCase):
    def modules(self):
        d,p=implementation('v13_data'),implementation('v13_protocol')
        self.assertIsNotNone(d,'V13 data protocol missing')
        self.assertIsNotNone(p,'V13 evaluation protocol missing')
        return d,p

    def test_exact_reconstruction_and_unaffected_facts(self):
        d,_=self.modules();a=d.make_probe(source(),'train',2)
        self.assertEqual(a,d.make_probe(source(),'train',2))
        self.assertEqual(len(a['facts'][0]),12)
        for c in range(4):
            mapping={tuple(x['key']):x['value'] for x in a['facts'][c]}
            for j,key in enumerate(a['query_keys']):self.assertEqual(mapping[tuple(key)],a['answers'][c][j])
        self.assertEqual(a['answers'][0][:2],list(reversed(a['answers'][1][:2])))
        self.assertEqual(a['answers'][0],a['answers'][2]);self.assertEqual(a['answers'][0],a['answers'][3])
        self.assertNotEqual(a['contexts'][0],a['contexts'][3])
        self.assertTrue(all(a['answers'][i][j]==a['answers'][0][j] for i in range(4) for j in (2,3)))

    def test_disjoint_entities_and_labels_without_unknown(self):
        d,_=self.modules();train=d.make_probe(source(),'train',4);dev=d.make_probe(source(),'development',4)
        self.assertTrue(set(train['entities']).isdisjoint(dev['entities']))
        t={f['value'] for c in train['facts'] for f in c};v={f['value'] for c in dev['facts'] for f in c}
        self.assertTrue(t.isdisjoint(v));self.assertTrue(all(len(x)==4 and x.isdigit() for x in t|v))
        self.assertNotIn('UNKNOWN',str(train['answers'])+str(dev['answers']))
        self.assertEqual(len(train['facts'][0]),24)
        with self.assertRaises(ValueError):d.make_probe(source(),'confirmation',2)

    def test_natural_seven_fixed_rows(self):
        d,_=self.modules();a=d.make_natural(source())
        pairs=[(i,j) for i,row in enumerate(a['answers']) for j,v in enumerate(row) if v is not None]
        self.assertEqual(pairs,[(0,0),(0,1),(1,0),(1,1),(2,1),(3,0),(3,1)])
        self.assertNotIn('UNKNOWN',[x for row in a['answers'] for x in row if x is not None])

    def test_interference_adds_facts_without_changing_existing_facts(self):
        d,_=self.modules();small=d.make_probe(source(),'train',2);large=d.make_probe(source(),'train',4)
        teams=set(small['entities'])
        for i in range(4):
            left={tuple(f['key']):f['value'] for f in small['facts'][i]}
            right={tuple(f['key']):f['value'] for f in large['facts'][i] if f['key'][0] in teams}
            self.assertEqual(left,right)

    def test_complete_correctness_not_identical_wrong_answers(self):
        d,p=self.modules();a=d.make_probe(source(),'train',2)
        rows=p.prediction_rows(a,[list(x) for x in a['answers']])
        m=p.metrics(rows);self.assertEqual(m['counts']['target_joint'],[1,1]);self.assertEqual(m['counts']['locality_joint'],[1,1])
        wrong=[dict(x,prediction='0000') for x in rows];self.assertEqual(p.metrics(wrong)['counts']['locality_joint'],[0,1])
        with self.assertRaises(ValueError):p.metrics(rows[:-1])
        with self.assertRaises(ValueError):p.metrics(rows+[rows[0]])

    def test_natural_pair_ignores_deleted_evidence_only(self):
        d,p=self.modules();a=d.make_natural(source())
        rows=p.prediction_rows(a,[list(x) for x in a['answers']]);m=p.metrics(rows)
        self.assertEqual(m['rows'],7);self.assertEqual(m['counts']['target_joint'],[1,1])
        self.assertEqual(m['counts']['locality_joint'],[1,1])
        bad=[dict(x,prediction='UNKNOWN') for x in rows]
        self.assertEqual(p.metrics(bad)['counts']['known_exact'],[0,7])

    def test_training_admits_only_fixed_family_counts_and_development(self):
        m=implementation('v13_training');self.assertIsNotNone(m,'V13 training entry missing')
        train=m.build_units([source('a'),source('b')],'train')
        dev=m.build_units([source('c')],'development')
        self.assertEqual(len(train),4);self.assertEqual(len(dev),4)
        self.assertEqual([u['level'] for u in dev if u['family']=='probe'],[2,3,4])
        with self.assertRaises(ValueError):m.build_units([source()],'confirmation')

    def test_learning_rate_decays_after_warmup_with_positive_floor(self):
        _,p=self.modules();self.assertAlmostEqual(p.learning_rate(32),5e-5)
        self.assertAlmostEqual(p.learning_rate(512),5e-6)
        self.assertGreater(p.learning_rate(128),p.learning_rate(256))
        with self.assertRaises(ValueError):p.learning_rate(1025,1025)

if __name__=='__main__':unittest.main()
