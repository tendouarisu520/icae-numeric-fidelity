import copy
import importlib
import unittest


def module():
    return importlib.import_module('binding.v11_pipeline')


class Tokenizer:
    bos_token_id=1
    eos_token_id=2
    def encode(self,text,add_special_tokens=False):
        return [3+sum(map(ord,x)) for x in text.split()]


class PipelineTests(unittest.TestCase):
    def test_contexts_are_encoded_without_questions_or_gold(self):
        m=module();u=dict(id='x',contexts=['Alpha.']*4,queries=['Who?']*3,
                         candidates=['Joe','Bob','UNKNOWN'],unaffected_answer='Sue')
        p=m.prepare_unit(u,Tokenizer())
        changed=copy.deepcopy(u);changed['queries']=['Where?']*3
        changed['candidates']=['Tom','Lee','UNKNOWN'];changed['unaffected_answer']='Alice'
        self.assertEqual(p['contexts'],m.prepare_unit(changed,Tokenizer())['contexts'])
        self.assertEqual(p['candidates'][0][-1],2)
        self.assertEqual(len(p['contexts'][0]),2)
        u['contexts'][0]='a '*512
        with self.assertRaises(ValueError):m.prepare_unit(u,Tokenizer())

    def test_metrics_separate_consistency_from_correctness_and_refusal(self):
        m=module();u=dict(id='u',target_answers=['A','B','UNKNOWN','A'],unaffected_answer='C')
        gold=[['A','C','UNKNOWN'],['B','C','UNKNOWN'],['UNKNOWN','C','UNKNOWN'],['A','C','UNKNOWN']]
        rows=m.prediction_rows(u,gold)
        result=m.metrics(rows)
        self.assertEqual(result['counts']['target_all_four'],[1,1])
        self.assertEqual(result['counts']['known_exact'],[7,7])
        self.assertEqual(result['counts']['unknown_exact'],[5,5])
        bad=copy.deepcopy(gold)
        for values in bad:values[1]='Wrong'
        bad[1][0]='A';bad[0][0]='UNKNOWN'
        result=m.metrics(m.prediction_rows(u,bad))
        self.assertEqual(result['counts']['unaffected_consistent'],[1,1])
        self.assertEqual(result['counts']['unaffected_all_four'],[0,1])
        self.assertEqual(result['counts']['known_false_unknown'],[1,7])
        self.assertEqual(result['counts']['known_target_misbinding'],[1,3])

    def test_no_unknown_credit_for_empty_or_explanatory_output(self):
        m=module()
        self.assertEqual(m.em_f1('UNKNOWN.','UNKNOWN'),(1,1.))
        self.assertEqual(m.em_f1('','UNKNOWN'),(0,0.))
        self.assertEqual(m.em_f1('The answer is UNKNOWN','UNKNOWN'),(0,0.))
        self.assertEqual(m.em_f1('The red car.','red car'),(1,1.))

    def test_selection_has_article_caps_and_is_deterministic(self):
        m=module();rows=[dict(id=str(i),article='a' if i<5 else 'b') for i in range(9)]
        selected=m.balanced_select(rows,4,2)
        self.assertEqual([r['id'] for r in selected],['0','5','1','6'])
        with self.assertRaises(ValueError):m.balanced_select(rows,5,2)


if __name__=='__main__':unittest.main(verbosity=2)
