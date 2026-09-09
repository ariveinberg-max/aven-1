"""Standalone arithmetic tool-use prototype for Aven-1.

Problem this addresses (see WRITEUP.md, "Small models memorize arithmetic;
they don't compute it"): the fine-tuned 58M model pattern-matches arithmetic
questions against memorized training examples instead of actually computing
the answer. Tested live, it answered "whats 1+1" with "1 plus 1 is 12." --
wrong. This module is a tool-use fallback: detect an arithmetic question in
the raw user message, compute the real answer with Python arithmetic, and
format it in the same style the model was trained to produce (see
make_instructions.py's ARITH_*_PHRASES and arithmetic_examples()), so the
correct answer is indistinguishable in style from the model's own voice.

This is a standalone proof-of-concept only. It is NOT wired into server.py
or chat.html.
"""

import re
from fractions import Fraction

# Word forms used both to detect an operation in formal phrasing and to
# render the answer in the same style as arithmetic_examples() in
# make_instructions.py ("{a} plus {b} is {a+b}.", etc).
_OP_WORDS = {
    'plus': '+',
    'minus': '-',
    'times': '*',
    'multiplied by': '*',
    'divided by': '/',
}

_WORD_FOR_OP = {
    '+': 'plus',
    '-': 'minus',
    '*': 'times',
    '/': 'divided by',
}

_NUM = r'-?\d+(?:\.\d+)?'

# Compact-notation patterns, e.g. "1+1", "6*7", "10 / 2". Symbol order
# matches ARITH_ADD_PHRASES / ARITH_SUB_PHRASES / ARITH_MUL_PHRASES.
_COMPACT_PATTERNS = [
    (re.compile(rf'({_NUM})\s*\+\s*({_NUM})'), '+'),
    (re.compile(rf'({_NUM})\s*-\s*({_NUM})'), '-'),
    (re.compile(rf'({_NUM})\s*[*x×]\s*({_NUM})'), '*'),
    (re.compile(rf'({_NUM})\s*/\s*({_NUM})'), '/'),
]

# Formal phrasing patterns, e.g. "What is 6 plus 7?", "Add 3 and 4.",
# "Subtract 4 from 9.", "Multiply 5 and 6.".
_WORD_OP_PATTERN = re.compile(
    rf'({_NUM})\s*(plus|minus|times|multiplied by|divided by)\s*({_NUM})',
    re.IGNORECASE,
)
_ADD_AND_PATTERN = re.compile(rf'\badd\s+({_NUM})\s+and\s+({_NUM})', re.IGNORECASE)
_MUL_AND_PATTERN = re.compile(rf'\bmultiply\s+({_NUM})\s+and\s+({_NUM})', re.IGNORECASE)
_SUB_FROM_PATTERN = re.compile(rf'\bsubtract\s+({_NUM})\s+from\s+({_NUM})', re.IGNORECASE)


def _to_number(s):
    f = Fraction(s)
    return f


def _format_number(n):
    if isinstance(n, Fraction):
        if n.denominator == 1:
            return str(n.numerator)
        # Fall back to a decimal rendering for non-integer results.
        return str(round(float(n), 6)).rstrip('0').rstrip('.')
    return str(n)


def detect_arithmetic(message):
    """Detect an arithmetic request and extract (a, op, b) as (Fraction, str, Fraction).

    Returns None if the message does not look like a basic arithmetic
    computation (addition, subtraction, multiplication, division).
    """
    if not message or not isinstance(message, str):
        return None

    text = message.strip()

    # "Subtract B from A" -> A - B (word order is reversed vs. surface order).
    m = _SUB_FROM_PATTERN.search(text)
    if m:
        b, a = m.group(1), m.group(2)
        return (_to_number(a), '-', _to_number(b))

    # "Add A and B"
    m = _ADD_AND_PATTERN.search(text)
    if m:
        return (_to_number(m.group(1)), '+', _to_number(m.group(2)))

    # "Multiply A and B"
    m = _MUL_AND_PATTERN.search(text)
    if m:
        return (_to_number(m.group(1)), '*', _to_number(m.group(2)))

    # "A plus/minus/times/divided by B" (formal word phrasing)
    m = _WORD_OP_PATTERN.search(text)
    if m:
        a, op_word, b = m.group(1), m.group(2).lower(), m.group(3)
        return (_to_number(a), _OP_WORDS[op_word], _to_number(b))

    # Compact symbol notation: "1+1", "6*7", "10/2", etc. Checked last so
    # word-based phrasing (which may itself contain no symbols) is preferred
    # when both could match, and so a plain word like "opposite" never
    # accidentally matches a symbol pattern (it can't, but scanning order
    # keeps intent clear).
    for pattern, op in _COMPACT_PATTERNS:
        m = pattern.search(text)
        if m:
            return (_to_number(m.group(1)), op, _to_number(m.group(2)))

    return None


def compute(a, op, b):
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '*':
        return a * b
    if op == '/':
        if b == 0:
            return None
        return a / b
    raise ValueError(f'unsupported operator: {op}')


def format_answer(a, op, b, result):
    """Render the answer in the same style as make_instructions.py's
    arithmetic_examples(): '{a} plus {b} is {a+b}.' etc.
    """
    a_str = _format_number(a)
    b_str = _format_number(b)
    if result is None:
        return f"{a_str} {_WORD_FOR_OP[op]} {b_str} is undefined (division by zero)."
    result_str = _format_number(result)
    return f"{a_str} {_WORD_FOR_OP[op]} {b_str} is {result_str}."


def answer_arithmetic(message):
    """Main entry point: given a raw user message, return the correct
    arithmetic answer string if the message is an arithmetic request,
    else None.
    """
    parsed = detect_arithmetic(message)
    if parsed is None:
        return None
    a, op, b = parsed
    result = compute(a, op, b)
    return format_answer(a, op, b, result)


if __name__ == '__main__':
    # Quick manual demo of the exact failing case from tonight's testing.
    failing_case = "whats 1+1"
    print(f"Input: {failing_case!r}")
    print(f"Model said (tonight, wrong): '1 plus 1 is 12.'")
    print(f"Tool says (correct):         '{answer_arithmetic(failing_case)}'")
