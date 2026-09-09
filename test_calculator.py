"""Tests for tools/calculator.py, the standalone arithmetic tool-use prototype.

Run with: python3 test_calculator.py
"""

from tools.calculator import answer_arithmetic


def check(label, message, expected):
    actual = answer_arithmetic(message)
    status = "PASS" if actual == expected else "FAIL"
    print(f"[{status}] {label}: {message!r} -> {actual!r}")
    assert actual == expected, f"{label}: expected {expected!r}, got {actual!r}"


def check_none(label, message):
    actual = answer_arithmetic(message)
    status = "PASS" if actual is None else "FAIL"
    print(f"[{status}] {label}: {message!r} -> {actual!r}")
    assert actual is None, f"{label}: expected None (not detected), got {actual!r}"


def main():
    # Formal phrasing (matches ARITH_ADD_PHRASES / ARITH_SUB_PHRASES / ARITH_MUL_PHRASES)
    check("formal addition", "What is 6 plus 7?", "6 plus 7 is 13.")
    check("formal subtraction", "What is 9 minus 4?", "9 minus 4 is 5.")
    check("formal multiplication", "What is 5 times 6?", "5 times 6 is 30.")
    check("add-and phrasing", "Add 3 and 4.", "3 plus 4 is 7.")
    check("subtract-from phrasing", "Subtract 4 from 9.", "9 minus 4 is 5.")
    check("multiply-and phrasing", "Multiply 5 and 6.", "5 times 6 is 30.")

    # Compact notation
    check("compact addition", "1+1", "1 plus 1 is 2.")
    check("compact addition casual", "whats 1+1", "1 plus 1 is 2.")
    check("compact multiplication", "6*7", "6 times 7 is 42.")
    check("compact subtraction", "10-3", "10 minus 3 is 7.")
    check("compact division", "10/2", "10 divided by 2 is 5.")
    check("compact x-multiplication", "6x7", "6 times 7 is 42.")

    # The exact failing case from tonight's testing: model said
    # "1 plus 1 is 12." (wrong) for input "whats 1+1".
    check("tonight's failing case", "whats 1+1", "1 plus 1 is 2.")

    # Negative case: should NOT be detected as arithmetic.
    check_none("non-arithmetic question", "what's the opposite of hot")
    check_none("non-arithmetic question 2", "Spell the word cat backwards.")
    check_none("empty string", "")

    print("\nAll tests passed.")


if __name__ == '__main__':
    main()
