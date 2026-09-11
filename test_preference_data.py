import copy
import unittest
from preference_data import split_comparisons, normalized


def comparison(prompt, a='Good answer', b='Bad answer', winner='a'):
    return dict(prompt=prompt, response_a=a, response_b=b, winner=winner)


class PreferenceDataTests(unittest.TestCase):
    def test_no_prompt_leakage_or_input_changes(self):
        rows = [comparison(f'Prompt {i}', a=f'Answer {j}') for i in range(4) for j in range(3)]
        original = copy.deepcopy(rows)
        train, val, report = split_comparisons(rows)
        train_prompts = {normalized(r['prompt']).casefold() for r in train}
        val_prompts = {normalized(r['prompt']).casefold() for r in val}
        self.assertFalse(train_prompts & val_prompts)
        self.assertEqual(rows, original)
        self.assertEqual(split_comparisons(rows), (train, val, report))

    def test_duplicate_reversal_and_conflicting_labels(self):
        rows = [comparison(str(i)) for i in range(5)]
        rows += [comparison('0', a='Bad answer', b='Good answer', winner='b')]
        rows += [comparison('conflict'), comparison('conflict', winner='b')]
        rows += [comparison('identical', a='same', b='same')]
        train, val, report = split_comparisons(rows)
        self.assertEqual(report['duplicates'], 1)
        self.assertEqual(report['conflicting'], 2)
        self.assertEqual(report['identical'], 1)
        self.assertEqual(len(train) + len(val), 5)
        self.assertFalse(any(r['prompt'] == 'conflict' for r in train + val))

    def test_one_prompt_cannot_produce_independent_validation(self):
        with self.assertRaisesRegex(ValueError, 'two prompts'):
            split_comparisons([comparison('same', a=str(i)) for i in range(10)])
