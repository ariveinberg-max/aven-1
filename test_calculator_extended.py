import unittest
from tools.calculator import answer_arithmetic


class CalculatorTests(unittest.TestCase):
    def test_existing_examples(self):
        from test_calculator import main
        main()

    def test_compound_expressions(self):
        for query, expected in [('2+3*4', '2+3*4 = 14.'),
                                ('(2+3)*4', '(2+3)*4 = 20.'),
                                ('Calculate 10 divided by (2 plus 3)', '10 / (2 + 3) = 2.'),
                                ('-3*-2', '-3 times -2 is 6.'),
                                ('0.1+0.2', '1/10 plus 1/5 is 3/10.'),
                                ('1/3', '1 divided by 3 is 1/3.')]:
            with self.subTest(query=query):
                self.assertEqual(answer_arithmetic(query), expected)

    def test_no_partial_or_executable_requests(self):
        for query in ['Explain why 2+2 is 4', '2026-09-09 meeting',
                      '2+3 and 4+5', '2**1000000', '__import__("os")',
                      '1//2', '2+2 apples', '2+', '9'*513]:
            with self.subTest(query=query):
                self.assertIsNone(answer_arithmetic(query))

    def test_zero_in_nested_expression(self):
        self.assertIn('undefined', answer_arithmetic('1/(2-2)'))
