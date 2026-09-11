import unittest
from prompting import build_chat_prompt


class PromptTests(unittest.TestCase):
    def test_latest_instruction_never_truncated(self):
        messages = [{'role': 'user', 'content': 'x' * 1000}]
        with self.assertRaisesRegex(ValueError, 'Shorten'):
            build_chat_prompt(messages, None, 128)

    def test_only_complete_recent_turns(self):
        messages = [{'role':'user','content':'old ' * 60}, {'role':'assistant','content':'old answer'},
                    {'role':'user','content':'recent'}, {'role':'assistant','content':'recent answer'},
                    {'role':'user','content':'latest'}]
        prompt, info = build_chat_prompt(messages, None, 180)
        self.assertNotIn('old answer', prompt)
        self.assertIn('recent answer', prompt)
        self.assertTrue(prompt.endswith('latest\n\n### Response:\n'))
        self.assertEqual(info['retained_turns'], 1)
        self.assertEqual(info['omitted_messages'], 2)
        self.assertLessEqual(info['prompt_tokens'], 180)

    def test_budget_uses_tokenizer(self):
        class Compressed:
            def encode(self, text): return list(range(len(text) // 4))
        prompt, info = build_chat_prompt([{'role':'user','content':'x'*200}], Compressed(), 100)
        self.assertEqual(info['prompt_tokens'], len(prompt)//4)

    def test_orphan_answers_and_failed_messages_omitted(self):
        messages = [{'role':'assistant','content':'orphan'}, {'role':'user','content':'failed'},
                    {'role':'user','content':'latest'}]
        prompt, info = build_chat_prompt(messages, None, 128)
        self.assertNotIn('orphan', prompt)
        self.assertNotIn('failed', prompt)
        self.assertEqual(info['omitted_messages'], 2)
