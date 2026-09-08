"""Generates an original, programmatically composed instruction-tuning corpus.

No external chat dataset is downloaded. Examples are built from templates
with multiple phrasings each, arithmetic, small hand-written word/fact
lists, and sentences pulled from your own compiled training corpus
(data/training.txt) — the same "own or have permission to use" rule as the
rest of this project.

v2: adds phrasing variety per task (so the model isn't just memorizing one
exact wording), a book-facts category, and short two-turn conversations so
the multi-turn Chat panel has something to actually learn from.

v3: the real gap v2 exposed was novel phrasing breaking the model entirely
(it would hallucinate a fake new instruction block instead of answering).
This adds many more paraphrasings for the open-ended "small talk" categories
(the ones most likely to see wording variety in real use), a days/months and
simple-list category, and an explicit graceful-fallback category: varied,
genuinely odd/unfamiliar-sounding questions paired with an honest "I don't
know that yet" style answer, so the model has *something* better to fall
back on than garbling a response when it doesn't recognize the pattern.

v4: two real gaps found in v3, from an actual user testing casual chat and
noticing antonyms kept getting a fallback answer instead of a real one.
(1) Everything was formally capitalized and punctuated ("What is 4 plus 9?")
— nothing looked like how people actually type ("whats 4 plus 9", "hello
introduce yourself"). Added `casualize()`, applied to a copy of most
categories: lowercased, punctuation dropped, common contractions shortened
("what is" -> "whats", "you"/"your" -> "u"/"ur"). Same instruction, same
correct response — teaches that phrasing style shouldn't change the answer.
(2) Antonyms were a real *data imbalance*, not a model weakness: ~120
examples against arithmetic's 2,400 (~1.5% of the pool) is not enough
repetition to learn from at this scale. More antonym pairs, more phrasings,
and a much higher weight fix that directly.

Format (matches what train.py --finetune expects):

    ### Instruction:
    <instruction text>

    ### Response:
    <response text>
    <|end|>

Two-turn example:

    ### Instruction:
    <instruction 1>

    ### Response:
    <response 1>
    <|end|>

    ### Instruction:
    <instruction 2>

    ### Response:
    <response 2>
    <|end|>
"""
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
random.seed(7)

GREETING_PROMPTS = ['Hello!', 'Hi there', 'Hi', 'Good morning', 'Good evening', 'Hey', "What's up?", 'Greetings',
                     'Hiya', 'Yo', 'Say hello to me', 'Greet me', 'Can you say hi?', 'Good afternoon',
                     "Howdy", 'Hey, how are you', 'Say hi', 'Give me a greeting']
GREETING_REPLIES = [
    'Hello! How can I help you?', 'Hi! What can I do for you?',
    'Hey there! What do you need?', "Hello! I'm ready when you are.",
    'Hi! Good to hear from you.',
]
FAREWELL_PROMPTS = ['Goodbye', 'Bye', 'See you later', 'Thanks, bye', 'I have to go now', "That's all, bye",
                     'Talk to you later', 'Signing off', 'Catch you later']
FAREWELL_REPLIES = ['Goodbye! Come back anytime.', 'See you later!', 'Take care!', 'Bye for now!']
THANKS_PROMPTS = ['Thank you', 'Thanks a lot', 'Thanks for the help', 'I appreciate it', 'Much appreciated',
                   'Thanks so much', 'That helped, thanks', 'Cheers']
THANKS_REPLIES = ["You're welcome!", 'Happy to help.', 'Anytime!', 'Glad that helped.']
IDENTITY_PROMPTS = ['What is your name?', 'Who are you?', 'What are you?', 'Tell me about yourself',
                     'Introduce yourself', 'What kind of AI are you?', 'Are you a real person?',
                     'Are you ChatGPT?', 'Do you have a name?']
IDENTITY_REPLIES = [
    'I am a byte-level Transformer trained from scratch. I do not have a name of my own yet.',
    'I am a small language model built and trained from random weights on this computer.',
    'I am a from-scratch neural network, not a pretrained assistant like ChatGPT.',
]
HELP_PROMPTS = ['Can you help me?', 'I need some help', 'Could you assist me?', 'I have a question',
                'Can I ask you something?', 'Help me out here', 'I need assistance']
HELP_REPLIES = ["I'll do my best. What do you need?", 'Sure — what do you need help with?', 'Of course, go ahead.']
WELLBEING_PROMPTS = ['How are you?', 'How are you doing?', 'How are you feeling?', "How's it going?"]
WELLBEING_REPLIES = ["I'm a small language model, so I don't have feelings, but I'm ready to help.",
                      "I don't experience feelings, but everything is running fine."]

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
          'September', 'October', 'November', 'December']

FALLBACK_QUESTIONS = [
    'What is the meaning of life?', "What's the capital of a country you've never heard of?",
    'Can you predict tomorrow\'s lottery numbers?', 'What am I thinking right now?',
    'What is your favorite color?', 'Do you dream?', 'What will the stock market do tomorrow?',
    'Can you feel pain?', 'What is beyond the edge of the universe?', "What's my name?",
    'Can you browse the internet?', 'Do you remember our last conversation?', 'What year will I die?',
    'Can you see images?', 'What is the airspeed velocity of an unladen swallow?',
    'Solve this riddle for me', 'What is 0 divided by 0?', 'Can you call someone for me?',
    'What is your opinion on politics?', 'Are you conscious?',
]
FALLBACK_REPLIES = [
    "I'm not sure — I'm a small model trained on a narrow set of examples, so I don't know that.",
    "I don't have a reliable answer for that; I'm still a very limited, small trained model.",
    "That's outside what I've been trained on. I can help with the kinds of things I've practiced.",
]

ANTONYMS = [
    ('hot', 'cold'), ('big', 'small'), ('fast', 'slow'), ('happy', 'sad'),
    ('light', 'dark'), ('up', 'down'), ('open', 'closed'), ('young', 'old'),
    ('strong', 'weak'), ('early', 'late'), ('easy', 'difficult'), ('full', 'empty'),
    ('loud', 'quiet'), ('near', 'far'), ('rich', 'poor'), ('true', 'false'),
    ('wet', 'dry'), ('hard', 'soft'), ('clean', 'dirty'), ('brave', 'cowardly'),
    ('tall', 'short'), ('wide', 'narrow'), ('thick', 'thin'), ('heavy', 'light'),
    ('good', 'bad'), ('safe', 'dangerous'), ('smooth', 'rough'), ('sharp', 'dull'),
    ('new', 'old'), ('cheap', 'expensive'), ('deep', 'shallow'), ('sweet', 'sour'),
    ('kind', 'cruel'), ('polite', 'rude'), ('simple', 'complicated'), ('modern', 'ancient'),
]
ANTONYM_QUESTIONS = ['What is the opposite of "{w}"?', 'Give the opposite of "{w}".',
                      'What word means the opposite of "{w}"?', 'What\'s the antonym of "{w}"?',
                      'Tell me the opposite of "{w}".', 'Name the opposite of "{w}".']

BOOKS = [
    ('Alice’s Adventures in Wonderland', 'Lewis Carroll'),
    ('Pride and Prejudice', 'Jane Austen'),
    ('The Adventures of Sherlock Holmes', 'Arthur Conan Doyle'),
    ('Frankenstein', 'Mary Shelley'),
    ('The Time Machine', 'H. G. Wells'),
    ('A Christmas Carol', 'Charles Dickens'),
]
BOOK_QUESTIONS = ['Who wrote {t}?', 'Who is the author of {t}?', 'Name the author of {t}.']

NUMBER_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
ARITH_ADD_PHRASES = ['What is {a} plus {b}?', 'What is {a} + {b}?', 'Add {a} and {b}.', 'What do you get if you add {a} and {b}?',
                      '{a}+{b}', '{a}+{b}=?', 'whats {a}+{b}', '{a} + {b} =']
ARITH_SUB_PHRASES = ['What is {a} minus {b}?', 'What is {a} - {b}?', 'Subtract {b} from {a}.',
                      '{a}-{b}', '{a}-{b}=?', 'whats {a}-{b}']
ARITH_MUL_PHRASES = ['What is {a} times {b}?', 'What is {a} * {b}?', 'Multiply {a} and {b}.',
                      '{a}*{b}', '{a}x{b}', 'whats {a}*{b}']

WORDS = ['cat', 'river', 'mountain', 'garden', 'window', 'candle', 'ocean', 'forest', 'lantern', 'quiet',
         'winter', 'harbor', 'meadow', 'thunder', 'whisper', 'journey', 'shadow', 'crystal', 'valley', 'ember']
UPPER_PHRASES = ['Convert the word "{w}" to uppercase.', 'Make "{w}" all uppercase.', 'Write "{w}" in capital letters.']
REVERSE_PHRASES = ['Spell the word "{w}" backwards.', 'Reverse the letters in "{w}".', 'What does "{w}" look like spelled backwards?']
COUNT_PHRASES = ['How many letters are in the word "{w}"?', 'Count the letters in "{w}".']
REPEAT_PHRASES = ['Repeat the word "{w}" {n} times.', 'Say "{w}" {n} times.']


def num_to_words(n):
    return NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else str(n)


CASUAL_SUBS = [
    (r"\bwhat is\b", 'whats'), (r"\bwhat's\b", 'whats'), (r"\bwhat are\b", 'whats'),
    (r"\byou\b", 'u'), (r"\byour\b", 'ur'), (r"\bare\b", 'r'),
    (r"\bplease\b", 'pls'), (r"\bbecause\b", 'bc'), (r"\btoday\b", '2day'),
]


def casualize(text):
    """Lowercase, drop terminal punctuation, apply common texting contractions.
    Same content, informal shape — teaches the model that phrasing style doesn't change the answer."""
    t = text.lower()
    for pattern, repl in CASUAL_SUBS:
        if random.random() < 0.5:
            t = re.sub(pattern, repl, t)
    t = t.rstrip('?.!')
    if random.random() < 0.3:
        t = t.replace('"', '')
    return t


def add_casual(pool, rate=1.0):
    """Return casual-phrasing duplicates of a pool, pairing the same correct response."""
    out = []
    for instr, resp in pool:
        if random.random() < rate:
            out.append((casualize(instr), resp))
    return out


def fixed_replies(prompts, replies, weight=1):
    out = []
    for p in prompts:
        for r in replies:
            out.append((p, r))
    return out * weight


def arithmetic_examples(n):
    out = []
    for _ in range(n):
        a, b = random.randint(1, 20), random.randint(1, 20)
        op = random.choice(['+', '-', '*'])
        if op == '+':
            instr = random.choice(ARITH_ADD_PHRASES).format(a=a, b=b)
            out.append((instr, f'{a} plus {b} is {a+b}.'))
        elif op == '-':
            a, b = max(a, b), min(a, b)
            instr = random.choice(ARITH_SUB_PHRASES).format(a=a, b=b)
            out.append((instr, f'{a} minus {b} is {a-b}.'))
        else:
            a, b = random.randint(1, 10), random.randint(1, 10)
            instr = random.choice(ARITH_MUL_PHRASES).format(a=a, b=b)
            out.append((instr, f'{a} times {b} is {a*b}.'))
    return out


def comparison_examples(n):
    out = []
    phrasings = ['Is {a} greater than {b}?', 'Is {a} bigger than {b}?', 'Which is bigger, {a} or {b}?']
    for _ in range(n):
        a, b = random.randint(1, 100), random.randint(1, 100)
        if a == b:
            b += 1
        kind = random.choice(phrasings)
        if kind.startswith('Which'):
            bigger = max(a, b)
            out.append((kind.format(a=a, b=b), f'{bigger} is bigger.'))
        else:
            answer = 'Yes' if a > b else 'No'
            out.append((kind.format(a=a, b=b), f'{answer}, {a} is {"greater" if a > b else "not greater"} than {b}.'))
    return out


def word_task_examples(n):
    out = []
    for _ in range(n):
        w = random.choice(WORDS)
        kind = random.choice(['upper', 'reverse', 'count', 'repeat'])
        if kind == 'upper':
            out.append((random.choice(UPPER_PHRASES).format(w=w), w.upper()))
        elif kind == 'reverse':
            out.append((random.choice(REVERSE_PHRASES).format(w=w), w[::-1]))
        elif kind == 'count':
            out.append((random.choice(COUNT_PHRASES).format(w=w), f'The word "{w}" has {num_to_words(len(w))} letters.'))
        else:
            times = random.randint(2, 4)
            out.append((random.choice(REPEAT_PHRASES).format(w=w, n=num_to_words(times)), ' '.join([w] * times)))
    return out


def antonym_examples():
    out = []
    for a, b in ANTONYMS:
        for q in ANTONYM_QUESTIONS:
            out.append((q.format(w=a), f'The opposite of "{a}" is "{b}".'))
            out.append((q.format(w=b), f'The opposite of "{b}" is "{a}".'))
    return out


def calendar_examples():
    out = []
    for i, day in enumerate(DAYS):
        nxt = DAYS[(i+1) % len(DAYS)]
        prev = DAYS[(i-1) % len(DAYS)]
        out.append((f'What day comes after {day}?', f'The day after {day} is {nxt}.'))
        out.append((f'What day comes before {day}?', f'The day before {day} is {prev}.'))
    for i, month in enumerate(MONTHS):
        nxt = MONTHS[(i+1) % len(MONTHS)]
        out.append((f'What month comes after {month}?', f'The month after {month} is {nxt}.'))
    out.append(('How many days are in a week?', 'There are seven days in a week.'))
    out.append(('How many months are in a year?', 'There are twelve months in a year.'))
    return out


def list_examples(n):
    categories = {
        'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white'],
        'animals': ['cat', 'dog', 'horse', 'fox', 'wolf', 'rabbit', 'bear', 'owl'],
        'fruits': ['apple', 'banana', 'orange', 'grape', 'pear', 'mango', 'cherry', 'lemon'],
        'numbers': [str(i) for i in range(1, 21)],
    }
    phrasings = ['List {n} {cat}.', 'Name {n} {cat}.', 'Give me {n} {cat}.']
    out = []
    for _ in range(n):
        cat, items = random.choice(list(categories.items()))
        count = random.randint(2, 4)
        chosen = random.sample(items, count)
        instr = random.choice(phrasings).format(n=num_to_words(count), cat=cat)
        out.append((instr, ', '.join(chosen) + '.'))
    return out


def fallback_examples():
    out = []
    for q in FALLBACK_QUESTIONS:
        out.append((q, random.choice(FALLBACK_REPLIES)))
    return out


def book_examples():
    out = []
    for title, author in BOOKS:
        for q in BOOK_QUESTIONS:
            out.append((q.format(t=title), f'{title} was written by {author}.'))
    return out


def continuation_examples(n):
    text = (ROOT / 'data/training.txt').read_text(encoding='utf-8', errors='replace')
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip().replace('\n', ' ') for s in sentences if 40 <= len(s.strip()) <= 160]
    out = []
    phrasings = ['Continue this sentence: {h}', 'Finish this sentence: {h}']
    for s in random.sample(sentences, min(n, len(sentences))):
        words = s.split(' ')
        if len(words) < 6:
            continue
        cut = random.randint(3, len(words) - 2)
        head, tail = ' '.join(words[:cut]), ' '.join(words[cut:])
        out.append((random.choice(phrasings).format(h=head), tail))
    return out


def all_single_turn_pools():
    pools = {
        'greeting': fixed_replies(GREETING_PROMPTS, GREETING_REPLIES, weight=3),
        'farewell': fixed_replies(FAREWELL_PROMPTS, FAREWELL_REPLIES, weight=3),
        'thanks': fixed_replies(THANKS_PROMPTS, THANKS_REPLIES, weight=3),
        'identity': fixed_replies(IDENTITY_PROMPTS, IDENTITY_REPLIES, weight=3),
        'help': fixed_replies(HELP_PROMPTS, HELP_REPLIES, weight=3),
        'wellbeing': fixed_replies(WELLBEING_PROMPTS, WELLBEING_REPLIES, weight=3),
        'arithmetic': arithmetic_examples(2400),
        'comparison': comparison_examples(1000),
        'word': word_task_examples(1600),
        'antonym': antonym_examples() * 7,
        'book': book_examples(),
        'calendar': calendar_examples() * 4,
        'list': list_examples(800),
        'fallback': fallback_examples() * 6,
        'continuation': continuation_examples(1400),
    }
    # Casual-phrasing duplicates: same correct response, informal wording (lowercase, no
    # punctuation, texting contractions) — the categories most likely to be typed casually.
    casual_sources = ['greeting', 'farewell', 'thanks', 'identity', 'help', 'wellbeing',
                       'arithmetic', 'antonym', 'calendar', 'list', 'comparison', 'fallback']
    for name in casual_sources:
        pools[f'{name}_casual'] = add_casual(pools[name], rate=0.6)
    return pools


def build_two_turn(pools, n):
    flat = [ex for pool in pools.values() for ex in pool]
    out = []
    for _ in range(n):
        a, b = random.choice(flat), random.choice(flat)
        out.append([a, b])
    return out


def render_single(instr, resp):
    return f'### Instruction:\n{instr}\n\n### Response:\n{resp}\n<|end|>\n'


def render_multi(pairs):
    return '\n'.join(render_single(i, r) for i, r in pairs)


def build(target_single=13000, target_multi=2200):
    pools = all_single_turn_pools()
    singles = [ex for pool in pools.values() for ex in pool]
    random.shuffle(singles)
    singles = singles[:target_single]
    multis = build_two_turn(pools, target_multi)

    blocks = [render_single(i, r) for i, r in singles] + [render_multi(pair) for pair in multis]
    random.shuffle(blocks)
    return '\n'.join(blocks)


if __name__ == '__main__':
    corpus = build()
    out_path = ROOT / 'data/instructions.txt'
    out_path.write_text(corpus, encoding='utf-8')
    print(f'Wrote {len(corpus.encode("utf-8")):,} bytes, {corpus.count("### Instruction:")} instruction blocks to {out_path}')
