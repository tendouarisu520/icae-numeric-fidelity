import copy
import importlib
import unittest


def module(test):
    try:value=importlib.import_module('binding.v11_data')
    except ModuleNotFoundError:value=None
    test.assertIsNotNone(value,'Natural relation-edit data builder missing')
    return value


def fixture():
    context=('Amelia Earhart was born in 1897. Charles Lindbergh was born in 1902. '
             'Wheat grows in broad fields during the mild summer season.')
    paragraph=dict(context=context,qas=[
        dict(id='q1',question='In which year was Amelia Earhart born?',is_impossible=False,answers=[dict(text='1897',answer_start=context.index('1897'))]),
        dict(id='q2',question='In which year was Charles Lindbergh born?',is_impossible=False,answers=[dict(text='1902',answer_start=context.index('1902'))]),
        dict(id='q3',question='In what year did Amelia Earhart win the Nobel Prize?',is_impossible=True,answers=[])])
    answers=[dict(text='1900',kind='time',title='Different article',id='a1',split='train')]
    donors=[dict(text='Quiet fields surround a remote and peaceful village.',title='Elsewhere',id='s1',split='train'),
            dict(text='An isolated observatory tracks small celestial objects during very clear nights.',title='Elsewhere',id='s2',split='train')]
    return paragraph,answers,donors


class DataTests(unittest.TestCase):
    def test_article_partition_is_deterministic(self):
        m=module(self)
        self.assertIn(m.article_split('One article'),('train','development','confirmation'))
        self.assertEqual(m.article_split('One article'),m.article_split('One article'))

    def test_four_edits_change_target_and_preserve_other_answer(self):
        m=module(self);paragraph,answers,donors=fixture()
        unit=m.make_unit('Aviation history',paragraph,answers,donors,'train')
        self.assertIsNotNone(unit)
        self.assertEqual(unit['candidates'],['1897','1900','UNKNOWN'])
        self.assertEqual(unit['target_answers'],['1897','1900','UNKNOWN','1897'])
        self.assertIn('Amelia Earhart was born in 1900.',unit['contexts'][1])
        self.assertNotIn('1897',unit['contexts'][2])
        self.assertTrue(all('Charles Lindbergh was born in 1902.' in context for context in unit['contexts']))
        self.assertEqual(unit['unaffected_answer'],'1902')
        self.assertEqual(unit['queries'][2],paragraph['qas'][2]['question'])
        self.assertTrue(m.audit_unit(unit))

    def test_repeated_target_and_overlapping_support_are_excluded(self):
        m=module(self);paragraph,answers,donors=fixture()
        paragraph['context']+=' The date 1897 is repeated.'
        self.assertIsNone(m.make_unit('Aviation history',paragraph,answers,donors,'train'))
        paragraph,answers,donors=fixture()
        paragraph['qas'][1]['answers']=[dict(text='Amelia Earhart',answer_start=0)]
        self.assertIsNone(m.make_unit('Aviation history',paragraph,answers,donors,'train'))

    def test_genuine_unanswerable_question_is_required(self):
        m=module(self);paragraph,answers,donors=fixture();paragraph['qas'].pop()
        self.assertIsNone(m.make_unit('Aviation history',paragraph,answers,donors,'train'))

    def test_donors_never_cross_partition_or_article(self):
        m=module(self);paragraph,answers,donors=fixture()
        for value in donors:value['split']='confirmation'
        self.assertIsNone(m.make_unit('Aviation history',paragraph,answers,donors,'train'))
        paragraph,answers,donors=fixture();answers[0]['title']='Aviation history'
        self.assertIsNone(m.make_unit('Aviation history',paragraph,answers,donors,'train'))

    def test_tampered_edit_cannot_pass_reconstruction_audit(self):
        m=module(self);paragraph,answers,donors=fixture()
        unit=m.make_unit('Aviation history',paragraph,answers,donors,'train')
        self.assertIsNotNone(unit)
        bad=copy.deepcopy(unit);bad['contexts'][2]+=' 1897'
        with self.assertRaises(ValueError):m.audit_unit(bad)
        bad=copy.deepcopy(unit);bad['target_answers'][2]='1912'
        with self.assertRaises(ValueError):m.audit_unit(bad)

    def test_abbreviations_do_not_create_dangling_sentence_fragments(self):
        m=module(self)
        text='St. John was born in 1900. Dr. Lee arrived in 1901.'
        self.assertEqual([text[a:b] for a,b in m.sentence_spans(text)],['St. John was born in 1900.','Dr. Lee arrived in 1901.'])

    def test_numeric_edit_keeps_units_and_plausible_scale(self):
        m=module(self)
        self.assertEqual(m.numeric_replacement('70%','83','number'),'83%')
        self.assertEqual(m.numeric_replacement('29 ways','25','number'),'25 ways')
        self.assertEqual(m.numeric_replacement('the 1960s','1970','time'),'the 1970s')
        self.assertEqual(m.numeric_replacement('1897','1900','time'),'1900')
        for a,b,kind in [('70%','120000','number'),('25','over 1000','number'),('1743','2007','time')]:
            self.assertIsNone(m.numeric_replacement(a,b,kind))

    def test_legal_and_biographic_abbreviations_are_not_sentence_breaks(self):
        m=module(self)
        text='The case was Poultry Corp. v. United States, 295 U.S. 495 (1935). The author died (d. 1908).'
        self.assertEqual([text[a:b] for a,b in m.sentence_spans(text)],
                         ['The case was Poultry Corp. v. United States, 295 U.S. 495 (1935).','The author died (d. 1908).'])

    def test_redundant_number_equations_are_not_edited(self):
        m=module(self);paragraph,answers,donors=fixture()
        context='The survey counted 10 birds including 4 adults. The leader was Amelia Earhart. Wheat grows in broad fields during the mild summer season.'
        paragraph=dict(context=context,qas=[
            dict(id='q1',question='How many birds were counted?',is_impossible=False,answers=[dict(text='10',answer_start=context.index('10'))]),
            dict(id='q2',question='Who was the leader?',is_impossible=False,answers=[dict(text='Amelia Earhart',answer_start=context.index('Amelia'))]),
            dict(id='q3',question='Who was the architect?',is_impossible=True,answers=[])])
        answers=[dict(text='12',kind='number',title='Elsewhere',id='a',split='train')]
        self.assertIsNone(m.make_unit('Aviation history',paragraph,answers,donors,'train'))


if __name__=='__main__':unittest.main(verbosity=2)
