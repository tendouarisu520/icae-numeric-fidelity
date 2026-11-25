import importlib
import unittest


def module(test):
    try:m=importlib.import_module('binding.v12_protocol')
    except ModuleNotFoundError:m=None
    test.assertIsNotNone(m,'Frozen-memory diagnostic protocol missing')
    return m


class ProtocolTests(unittest.TestCase):
    def test_shuffled_memories_are_bijective_and_article_disjoint(self):
        m=module(self);units=[dict(id=str(i),article=str(i//2)) for i in range(8)]
        a=m.shuffled_donors(units)
        self.assertEqual(a,m.shuffled_donors(list(reversed(units))))
        self.assertEqual(set(a),set(a.values()))
        by={u['id']:u['article'] for u in units}
        self.assertTrue(all(by[k]!=by[v] for k,v in a.items()))
        with self.assertRaises(ValueError):m.shuffled_donors([dict(id='a',article='x'),dict(id='b',article='x')])

    def test_sequence_ban_preserves_shared_prefix_words(self):
        m=module(self)
        self.assertEqual(m.banned_next([],[[5,6],[9]]),{9})
        self.assertEqual(m.banned_next([5],[[5,6],[9]]),{6})
        self.assertNotIn(7,m.banned_next([5],[[5,6]]))
        self.assertEqual(m.banned_next([1,5],[[5,6]]),set())

    def test_calibration_never_splits_an_article_and_recovers_a_threshold(self):
        m=module(self);records=[];articles={}
        for a in range(4):
            for j in range(2):
                uid=f'{a}-{j}';articles[uid]=str(a)
                records.append(dict(unit_id=uid,context_index=0,question_index=0,gold='answer' if j==0 else 'UNKNOWN',
                    prediction='UNKNOWN',alternative='answer',margin=2. if j==0 else -2.,alternative_valid=True))
        c=m.calibrate(records,articles)
        self.assertEqual(c['out_of_fold_f1'],1.)
        self.assertEqual(c['out_of_fold_predictions'],[r['gold'] for r in records])
        self.assertEqual(set(c['article_folds'].values()),{0,1})
        for f in c['folds']:
            self.assertFalse(set(f['fit_articles'])&set(f['evaluation_articles']))

    def test_invalid_alternative_stays_unknown_regardless_of_margin(self):
        m=module(self)
        self.assertEqual(m.prediction_at(dict(margin=100.,alternative='UNKNOWN',alternative_valid=False),-1.),'UNKNOWN')
        self.assertEqual(m.prediction_at(dict(margin=1.,alternative='answer',alternative_valid=True),1.),'UNKNOWN')


if __name__=='__main__':unittest.main()
