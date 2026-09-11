"""Local chat calculator with bounded, exact arithmetic and complete-request parsing."""
import ast
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
        return f'{n.numerator}/{n.denominator}'
    return str(n)


def expression_from_message(message):
    """Accept a complete calculation request, never a substring of prose."""
    if not isinstance(message, str) or len(message) > 512:
        return None
    text = message.strip().lower().rstrip('?.!').strip()
    text = re.sub(r"^(?:please\s+)?(?:what is|what's|whats|calculate|compute|evaluate)\s+", '', text)
    for pattern, op, reverse in ((_SUB_FROM_PATTERN, '-', True),
                                  (_ADD_AND_PATTERN, '+', False),
                                  (_MUL_AND_PATTERN, '*', False)):
        match = pattern.fullmatch(text)
        if match:
            a, b = match.groups()
            return f'{b} - {a}' if reverse else f'{a} {op} {b}'
    for word, op in _OP_WORDS.items():
        text = re.sub(r'\b' + word + r'\b', op, text)
    text = text.replace('×', '*').replace('÷', '/').replace('−', '-')
    text = re.sub(r'(?<=\d)\s*x\s*(?=[-+\d(])', '*', text)
    if not re.fullmatch(r'[0-9.\s()+*/-]+', text) or not any(c in text for c in '+-*/'):
        return None
    return text


def _parse(expression):
    tree = ast.parse(expression, mode='eval').body
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError('Expression too complex')
    return tree


def _evaluate(node, expression):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        result = Fraction(ast.get_source_segment(expression, node))
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        result = _evaluate(node.operand, expression)
        if isinstance(node.op, ast.USub):
            result = -result
    elif isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
        a, b = _evaluate(node.left, expression), _evaluate(node.right, expression)
        op = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}[type(node.op)]
        result = compute(a, op, b)
        if result is None:
            raise ZeroDivisionError
    else:
        raise ValueError('Unsupported expression')
    if max(result.numerator.bit_length(), result.denominator.bit_length()) > 4096:
        raise ValueError('Result too large')
    return result


def detect_arithmetic(message):
    """Compatibility API for complete two-operand calculations."""
    expression = expression_from_message(message)
    if expression is None:
        return None
    try:
        node = _parse(expression)
        if not isinstance(node, ast.BinOp) or type(node.op) not in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            return None
        if isinstance(node.left, ast.BinOp) or isinstance(node.right, ast.BinOp):
            return None
        return (_evaluate(node.left, expression),
                {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}[type(node.op)],
                _evaluate(node.right, expression))
    except (SyntaxError, ValueError, ZeroDivisionError):
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
    """Compute bounded arithmetic with precedence and exact decimal inputs.

    No eval, calls, names, powers, or partial matches. Unsupported requests
    return None so chat can continue through the language model.
    """
    expression = expression_from_message(message)
    if expression is None:
        return None
    try:
        result = _evaluate(_parse(expression), expression)
    except ZeroDivisionError:
        return f'{expression} is undefined (division by zero).'
    except (SyntaxError, ValueError, OverflowError, RecursionError):
        return None
    parsed = detect_arithmetic(message)
    if parsed is not None:
        return format_answer(*parsed, result)
    return f'{expression} = {_format_number(result)}.'
