import unittest
from recall import search_records, answer_recall, recall_query


class RecallTests(unittest.TestCase):
    records = [dict(id=1, title='Rocket project', content='Launch uses the blue engine.'),
               dict(id=2, title='Garden', content='Water the roses on Friday.'),
               dict(id=3, title='Shopping', content='Buy blue paint.')]

    def test_relevant_sources(self):
        result = answer_recall('/recall rocket engine', self.records)
        self.assertEqual([r['id'] for r in result['sources']], [1])
        self.assertIn('blue engine', result['reply'])
        self.assertEqual(result['tool'], 'memory')

    def test_absent_knowledge_not_invented(self):
        result = answer_recall('search my notes about quantum', self.records)
        self.assertEqual(result['sources'], [])
        self.assertIn('No matching', result['reply'])

    def test_explicit_recall_only(self):
        self.assertIsNone(recall_query('Hello, how are you?'))
        self.assertEqual(recall_query('What do you remember about roses?'), 'roses?')
        self.assertIn('Add a topic', answer_recall('/recall', self.records)['reply'])

    def test_excerpt_bounded_and_notes_unchanged(self):
        records = [dict(id=7, title='Long', content='x ' * 1000 + 'rocket engine')]
        original = records[0].copy()
        result = search_records('rocket', records)
        self.assertLessEqual(len(result[0]['excerpt']), 702)
        self.assertIn('rocket', result[0]['excerpt'])
        self.assertEqual(records[0], original)
