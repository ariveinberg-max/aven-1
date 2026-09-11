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
                     'Talk to you later', 'Signing off', 'Catch you later', 'See you later then', 'Alright, bye then']
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
WELLBEING_PROMPTS = ['How are you?', 'How are you doing?', 'How are you feeling?', "How's it going?",
                      'How are you doing today?', "How's it going today?"]
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

# v8: live testing (2026-09-10) found "can u help me with a math problem?" got a
# garbled reply blending a HELP_REPLIES fragment with a FALLBACK_REPLIES fragment --
# not a multi-turn-context bug (ruled out separately), but a genuine data gap:
# HELP_PROMPTS is only bare, topic-less requests ("Can you help me?"), and
# FALLBACK_QUESTIONS is only direct out-of-scope questions, so "help me with X"
# naming a specific task falls between both categories with nothing anchoring it.
# These pair that exact phrasing shape with the honest FALLBACK_REPLIES (the model
# genuinely can't help with most named tasks beyond what it was narrowly trained
# on), so the two "help" shapes point at two different, well-anchored replies
# instead of interpolating between them.
HELP_WITH_TOPIC_PROMPTS = [
    'Can you help me with a math problem?', 'Can you help me with my homework?',
    'Can you help me with something complicated?', 'Can you help me write an essay?',
    'Can you help me fix my code?', 'Can you help me plan a trip?',
    'Could you help me with something hard?', 'Can you help with a physics question?',
    'Can you help me with a science project?', 'Can you help me solve a puzzle?',
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

# v5: testing found the model has zero real-world factual knowledge -- it was
# only ever trained on synthetic arithmetic/antonym/greeting patterns, so a
# question like "who was the first president" gets a plausible-sounding but
# meaningless answer. These are hand-verified, timeless facts (no "current
# president" style facts that go stale) covering history, geography, and
# basic science -- a small, honest start, not a knowledge base.
FACTS = [
    ('Who was the first president of the United States?', 'George Washington was the first president of the United States.'),
    ('What is the capital of France?', 'The capital of France is Paris.'),
    ('What is the capital of Japan?', 'The capital of Japan is Tokyo.'),
    ('What is the capital of Italy?', 'The capital of Italy is Rome.'),
    ('What is the capital of Germany?', 'The capital of Germany is Berlin.'),
    ('What is the capital of the United Kingdom?', 'The capital of the United Kingdom is London.'),
    ('What is the capital of Russia?', 'The capital of Russia is Moscow.'),
    ('What is the capital of China?', 'The capital of China is Beijing.'),
    ('What is the capital of Canada?', 'The capital of Canada is Ottawa.'),
    ('What is the capital of Australia?', 'The capital of Australia is Canberra.'),
    ('What is the capital of Egypt?', 'The capital of Egypt is Cairo.'),
    ('How many continents are there?', 'There are seven continents.'),
    ('How many oceans are there?', 'There are five oceans.'),
    ('What is the largest planet in the solar system?', 'Jupiter is the largest planet in the solar system.'),
    ('What is the smallest planet in the solar system?', 'Mercury is the smallest planet in the solar system.'),
    ('Which planet is closest to the sun?', 'Mercury is the planet closest to the sun.'),
    ('Which planet do we live on?', 'We live on the planet Earth.'),
    ('At what temperature does water boil, in Celsius?', 'Water boils at 100 degrees Celsius at sea level.'),
    ('At what temperature does water freeze, in Celsius?', 'Water freezes at 0 degrees Celsius.'),
    ('How many legs does a spider have?', 'A spider has eight legs.'),
    ('How many legs does an insect have?', 'An insect has six legs.'),
    ('What is the chemical symbol for water?', 'The chemical symbol for water is H2O.'),
    ('What is the chemical symbol for gold?', 'The chemical symbol for gold is Au.'),
    ('What gas do humans need to breathe to survive?', 'Humans need oxygen to survive.'),
    ('What gas do plants absorb from the air?', 'Plants absorb carbon dioxide from the air.'),
    ('How many bones are in the adult human body?', 'An adult human body has 206 bones.'),
    ('What is the largest ocean on Earth?', 'The Pacific Ocean is the largest ocean on Earth.'),
    ('What is the longest river in the world?', 'The Nile is generally considered the longest river in the world.'),
    ('What is the tallest mountain in the world?', 'Mount Everest is the tallest mountain in the world.'),
    ('What is the largest desert in the world?', 'The Antarctic Desert is the largest desert in the world.'),
    ('Who painted the Mona Lisa?', 'Leonardo da Vinci painted the Mona Lisa.'),
    ('Who developed the theory of relativity?', 'Albert Einstein developed the theory of relativity.'),
    ('Who wrote Romeo and Juliet?', 'William Shakespeare wrote Romeo and Juliet.'),
    ('What year did World War II end?', 'World War II ended in 1945.'),
    ('What is the speed of light approximately, in kilometers per second?', 'The speed of light is approximately 300,000 kilometers per second.'),
    ('How many colors are in a rainbow?', 'A rainbow has seven colors.'),
    ('How many sides does a hexagon have?', 'A hexagon has six sides.'),
    ('How many sides does a triangle have?', 'A triangle has three sides.'),
    ('What is the freezing point of water in Fahrenheit?', 'Water freezes at 32 degrees Fahrenheit.'),
    ('What is the largest mammal on Earth?', 'The blue whale is the largest mammal on Earth.'),
    ('What is the currency used in the United States?', 'The currency used in the United States is the dollar.'),
    ('What is the currency used in Japan?', 'The currency used in Japan is the yen.'),
    # v9: live testing found Spain/Brazil (entirely absent before) produced hallucinated
    # garbage ("The capital of Gerry Wedneser.") instead of any recognizable fallback --
    # covering more of the handful of countries people actually ask about first.
    ('What is the capital of Spain?', 'The capital of Spain is Madrid.'),
    ('What is the capital of Brazil?', 'The capital of Brazil is Brasilia.'),
    ('What is the capital of Mexico?', 'The capital of Mexico is Mexico City.'),
    ('What is the capital of India?', 'The capital of India is New Delhi.'),
]
FACT_PHRASE_PREFIXES = ['{q}', 'Quick question: {q}', 'Do you know {q_lower}', 'Tell me, {q_lower}']


def fact_question_variants(q):
    q_lower = q[0].lower() + q[1:]
    return [p.format(q=q, q_lower=q_lower) for p in FACT_PHRASE_PREFIXES]

# v6: RLHF preference labeling stalled at 12 decided / 18 tied comparisons because
# the model has zero trained substance on open-ended/opinion prompts -- WRITEUP.md's
# RLHF section already diagnosed this exact failure mode for the old 19.8M model
# (creative-writing prompts came back "decisive but incoherent," never fixed by
# more phrasing variety since there was nothing real underneath). FACTS above fixed
# single-sentence factual gaps; this fixes multi-sentence opinion/explanation gaps
# the same way GREETING_REPLIES/IDENTITY_REPLIES fixed small talk: several genuinely
# different, coherent, hand-written 2-3 sentence answers per question, not one
# canned reply -- so sampling at different temperatures has real variance to label
# instead of ties (same answer every time) or noise (no trained answer at all).
OPEN_ENDED = [
    ("What makes a good friend?", [
        "A good friend listens without judging and tells you the truth even when it's hard to hear. They also show up when it actually matters, not just when it's convenient.",
        "I'd say trust matters most — someone who keeps their word and keeps your secrets. Beyond that, they support your decisions even when they'd choose differently themselves.",
        "Reliability is the real test: does this person show up on a bad day, not just a good one? Everything else, like shared interests, matters much less than that.",
    ]),
    ("Is it better to be cautious or take risks?", [
        "It depends on what's at stake. For something reversible, taking the risk usually teaches you more; for something permanent, caution protects you from a mistake you can't undo.",
        "I'd lean toward taking risks early in anything new, since the cost of a small failure is low and the information you gain is valuable. Caution matters more once a lot is already built on the outcome.",
        "Neither one is right on its own. Being too cautious means missing real opportunities, while being too reckless means the failures pile up faster than you can recover from them.",
    ]),
    ("Would you rather be invisible or be able to fly?", [
        "I'd pick flying. Invisibility is mostly useful for avoiding people, but flying would actually let you get somewhere and see the world differently.",
        "Invisibility, honestly — it seems more useful day to day, like avoiding an awkward conversation or watching something unfold without being noticed.",
        "Flying, without much hesitation. It solves a real, everyday problem, getting places, while invisibility is more of a novelty than something genuinely useful.",
    ]),
    ("What is more important, money or happiness?", [
        "Happiness is the actual goal, but money is often what removes obstacles in the way of it, like stress about bills or medical care. Past a certain point, though, more money stops adding much happiness at all.",
        "I'd say happiness, since money is only ever a means to something else. A lot of the things that make people happiest, close relationships, good health, don't cost much at all.",
        "It's not really a fair fight since one is a resource and the other is a state of mind. Enough money removes a lot of daily stress, but it doesn't guarantee happiness on its own.",
    ]),
    ("Is it better to work alone or with a team?", [
        "It depends on the task. Alone is faster for something with one clear right answer, while a team is better when the problem benefits from different perspectives catching each other's blind spots.",
        "I'd lean toward teams for most real work, since a second person catches mistakes you can't see in your own thinking. Working alone is more efficient, but only when nothing much can go wrong.",
        "Working alone gives you more control and fewer meetings, but a good team gets you further on anything complicated, since no one person has every skill a hard problem actually needs.",
    ]),
    ("What is the best way to learn something new?", [
        "Actually doing the thing beats reading about it almost every time, since mistakes you make yourself stick in a way that reading never quite does. Reading is best used to fill in gaps once you're already stuck.",
        "I'd say consistent small practice beats occasional long sessions. An hour a day for a month teaches more than one twelve-hour cram session, since the spacing helps it actually stick.",
        "Teaching it to someone else is one of the fastest ways to find out what you don't actually understand yet. Trying to explain something clearly exposes the gaps that just reading over it hides.",
    ]),
    ("Do you think technology makes life better or worse?", [
        "Mostly better, on balance — medicine, communication, and access to information have all improved enormously. It also creates new problems, like distraction and privacy loss, that didn't exist before.",
        "It's genuinely mixed. Technology solves real problems, like connecting people across distance, but it also introduces new ones, like the amount of attention it constantly asks for.",
        "I'd say it depends heavily on how it's used rather than the technology itself. The same phone that wastes someone's whole evening also lets them learn a new skill for free.",
    ]),
    ("What is more important, talent or hard work?", [
        "Hard work matters more over time, since talent without effort tends to plateau early while consistent effort keeps compounding. Talent mostly just decides how fast the first few steps go.",
        "I'd say they solve different problems: talent gives you a head start, but hard work is what actually gets you to a genuinely high level and keeps you there.",
        "Talent is overrated compared to how it gets talked about. Most people who look naturally gifted actually put in enormous, mostly invisible amounts of practice to get there.",
    ]),
    ("Should people always tell the truth, even if it hurts?", [
        "Mostly yes, since trust breaks down once people can't rely on what you tell them. There's still room for tact in how something true gets said, without changing what's actually true.",
        "I'd make room for small exceptions, like sparing someone's feelings over something trivial, but for anything that actually matters, the truth should come first.",
        "Honesty should be the default, but delivery matters just as much as the fact itself. A true thing said cruelly can do as much damage as a lie.",
    ]),
]
OPEN_ENDED_WRAPPERS = ['{q}', 'What do you think: {q_lower}', 'In your opinion, {q_lower}', "I'm curious, {q_lower}"]


def open_ended_examples():
    out = []
    for q, replies in OPEN_ENDED:
        q_lower = q[0].lower() + q[1:]
        for wrapper in OPEN_ENDED_WRAPPERS:
            wrapped = wrapper.format(q=q, q_lower=q_lower)
            for reply in replies:
                out.append((wrapped, reply))
    return out


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
    # v10: live testing found "which is bigger, 7 or 3?" answered with the WRONG
    # template's shape ("No, 14 is not greater than 7.") and an unrelated number --
    # the same combinatorial-sparsity problem already documented for arithmetic:
    # 1-100 gives ~10,000 possible pairs for only 1000 examples. Narrowed to 1-20,
    # matching arithmetic's honestly-trainable range, rather than adding more
    # examples at a range this model has no realistic chance of covering.
    out = []
    phrasings = ['Is {a} greater than {b}?', 'Is {a} bigger than {b}?', 'Which is bigger, {a} or {b}?']
    for _ in range(n):
        a, b = random.randint(1, 20), random.randint(1, 20)
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


# v11: Codex's independent capability-v1 eval scored 0% on comprehension,
# instruction-following, and code-reading -- categories this project's training
# data never covered at all (only arithmetic/facts/antonyms/small-talk existed).
# This is the first attempt at that real gap, not more RLHF/data-quality polish.
# Deliberately generates NEW instances of each skill (different names/values/code)
# rather than anything resembling capability-v1.json's frozen questions -- training
# on the actual eval questions would invalidate every future measurement against
# that suite, exactly the mistake capability_errors.py's own docs warn against.
COMP_NAMES = ['Priya', 'Jamal', 'Elena', 'Omar', 'Sofia', 'Liam', 'Ines', 'Kenji', 'Noor', 'Dana']
COMP_OBJECTS = ['key', 'ball', 'pen', 'coin', 'ring', 'marble', 'watch', 'stamp']
COMP_CONTAINERS = ['box', 'bag', 'drawer', 'jar', 'basket', 'crate']
COMP_COLORS = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange']


def comprehension_examples(n):
    out = []
    for _ in range(n):
        name1, name2 = random.sample(COMP_NAMES, 2)
        obj1, obj2 = random.sample(COMP_OBJECTS, 2)
        color1, color2 = random.sample(COMP_COLORS, 2)
        cont1, cont2 = random.sample(COMP_CONTAINERS, 2)
        cont_color1, cont_color2 = random.sample(COMP_COLORS, 2)
        story = (f'{name1} put a {color1} {obj1} in the {cont_color1} {cont1}. '
                 f'{name2} put a {color2} {obj2} in the {cont_color2} {cont2}.')
        if random.random() < 0.5:
            q = f'{story} What color is the {cont1} that has the {color1} {obj1}? Answer one word.'
            out.append((q, cont_color1))
        else:
            q = f'{story} Who put the {color2} {obj2} away? Answer one word.'
            out.append((q, name2))
    return out


INSTR_WORDS = ['amber', 'quartz', 'meadow', 'velvet', 'harbor', 'lantern', 'thicket', 'copper',
               'willow', 'granite', 'cinder', 'orchid']
INSTR_PHRASES = ['the quiet river', 'a sudden storm', 'three old maps', 'the last candle']
INSTRUCTION_FOLLOWING_TEMPLATES = [
    ('Write only the word {w}.', '{w}'),
    ('Reply with just the word {w}, nothing else.', '{w}'),
    ('Answer with exactly one word: {w}.', '{w}'),
    ('Say only "{w}".', '{w}'),
]


def instruction_following_examples(n):
    out = []
    for _ in range(n):
        if random.random() < 0.7:
            w = random.choice(INSTR_WORDS)
            template, answer = random.choice(INSTRUCTION_FOLLOWING_TEMPLATES)
            out.append((template.format(w=w), answer.format(w=w)))
        else:
            phrase = random.choice(INSTR_PHRASES)
            out.append((f'Repeat exactly: {phrase}', phrase))
    return out


def code_reading_examples(n):
    out = []
    for _ in range(n):
        kind = random.choice(['index', 'add', 'first_char', 'double', 'length'])
        if kind == 'index':
            vals = random.sample(range(1, 50), 3)
            i = random.randint(0, 2)
            code = f'x = [{vals[0]}, {vals[1]}, {vals[2]}]\nprint(x[{i}])'
            answer = str(vals[i])
        elif kind == 'add':
            a, b = random.randint(1, 30), random.randint(1, 30)
            code = f'x = {a}\ny = {b}\nprint(x + y)'
            answer = str(a + b)
        elif kind == 'first_char':
            w = random.choice(WORDS)
            code = f's = "{w}"\nprint(s[0])'
            answer = w[0]
        elif kind == 'double':
            a = random.randint(1, 40)
            code = f'x = {a}\nx = x * 2\nprint(x)'
            answer = str(a * 2)
        else:
            vals = random.sample(range(1, 50), random.choice([2, 3, 4]))
            code = f'lst = [{", ".join(map(str, vals))}]\nprint(len(lst))'
            answer = str(len(vals))
        out.append((f'Python:\n{code}\nWhat is printed? Output only the value.', answer))
    return out


def fact_examples():
    out = []
    for q, a in FACTS:
        for variant in fact_question_variants(q):
            out.append((variant, a))
    return out


def book_examples():
    out = []
    for title, author in BOOKS:
        for q in BOOK_QUESTIONS:
            out.append((q.format(t=title), f'{title} was written by {author}.'))
    return out


def continuation_examples(n, max_bytes=20_000_000):
    # Read only a bounded prefix, not the whole file: data/training.txt is now the
    # ~6.95GB Wikipedia+books corpus (was 354MB when this function was written), and
    # loading it whole risks the same macOS memory-pressure kill documented twice in
    # research/TASKS.md for other jobs on this 8GB Mac. A 20MB sample (same bound
    # sources.py/train.py already use for tokenizer training) still gives thousands
    # of diverse candidate sentences -- plenty for `n` in the low thousands.
    with open(ROOT / 'data/training.txt', 'r', encoding='utf-8', errors='replace') as f:
        text = f.read(max_bytes)
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
        'comparison': comparison_examples(1000) * 3,
        'word': word_task_examples(1600),
        'antonym': antonym_examples() * 7,
        'fact': fact_examples() * 5,
        'open_ended': open_ended_examples() * 5,
        # v9: live testing (2026-09-10) found this category getting spliced into an
        # unrelated FACTS answer ("who is the author of frankenstein" ->
        # "Franker freezes at 32 degrees Fahrenheit.") -- the exact same failure
        # shape as the original "George Water freezes..." bug, but happening in a
        # single fresh turn with no conversation history, which rules out the
        # multi-turn-context explanation. The real cause here: BOOK had only 18
        # examples at 1x weight versus FACTS's ~800 examples at 5x weight plus
        # casual-phrasing duplicates that BOOK never got at all -- a plain density
        # imbalance, not a context problem. Weighted up and added to casual_sources
        # below to match FACTS's treatment.
        'book': book_examples() * 5,
        # v10: live testing found "what day comes after Monday" answered with the
        # right format but the wrong day -- calendar is a small, finite, easily
        # memorizable set (7 days), the same shape as book, which weight alone fixed.
        'calendar': calendar_examples() * 8,
        'list': list_examples(800),
        'fallback': fallback_examples() * 6,
        'help_with_topic': fixed_replies(HELP_WITH_TOPIC_PROMPTS, FALLBACK_REPLIES, weight=4),
        'continuation': continuation_examples(1400),
        'comprehension': comprehension_examples(1200),
        'instruction_following': instruction_following_examples(800),
        'code_reading': code_reading_examples(1200),
    }
    # Casual-phrasing duplicates: same correct response, informal wording (lowercase, no
    # punctuation, texting contractions) — the categories most likely to be typed casually.
    casual_sources = ['greeting', 'farewell', 'thanks', 'identity', 'help', 'wellbeing',
                       'arithmetic', 'antonym', 'calendar', 'list', 'comparison', 'fallback', 'fact',
                       'open_ended', 'help_with_topic', 'book', 'comprehension', 'instruction_following']
    # code_reading deliberately excluded: casualize() would corrupt Python syntax.
    for name in casual_sources:
        pools[f'{name}_casual'] = add_casual(pools[name], rate=0.6)
    return pools


def build_multi_turn(pools, n, max_turns=4):
    # v7: real dashboard usage (2026-09-10 live testing) showed the model splicing
    # two unrelated memorized facts together mid-response when several short,
    # unrelated turns (greeting, a declined math request, wrong arithmetic) sat in
    # context before the real question -- e.g. "who was the first president" got
    # answered "George Water freezes at 0 degrees Celsius.", blending the Washington
    # fact with an unrelated water-freezing fact. The old build_two_turn() only ever
    # produced exactly 2 unrelated turns, so the model had almost no training
    # exposure to answering correctly with 3+ turns of unrelated clutter already in
    # its 192-token context -- the real shape of an actual chat session. This
    # generates 2-4 turn conversations instead of a fixed 2, still from randomly
    # unrelated pool entries (deliberately: the point is robustness to *whatever*
    # preceded the current turn, not topical continuity between turns).
    # NOT YET VERIFIED to reduce splicing -- that requires finetuning on the
    # regenerated corpus and re-testing the exact live prompts that exposed this,
    # which was deferred this session for lack of free memory on this machine (see
    # research/TASKS.md). Treat this as an untested hypothesis-driven data change
    # until that finetune runs.
    flat = [ex for pool in pools.values() for ex in pool]
    out = []
    for _ in range(n):
        turns = random.randint(2, max_turns)
        out.append([random.choice(flat) for _ in range(turns)])
    return out


def render_single(instr, resp):
    return f'### Instruction:\n{instr}\n\n### Response:\n{resp}\n<|end|>\n'


def render_multi(pairs):
    return '\n'.join(render_single(i, r) for i, r in pairs)


def build(target_single=26000, target_multi=4000):
    # v11: raised again (23000 -> 26000) after adding comprehension/
    # instruction_following/code_reading -- same truncation-dilution mistake
    # already found and fixed once this session if left at the old value.
    # v10: target_single was fixed at 13000 while the total single-turn pool grew to
    # ~19000-22000 across this session's additions, meaning the random shuffle-and-
    # truncate step was silently dropping ~30-40% of examples -- disproportionately
    # hurting small categories relative to giant ones like antonym/arithmetic. Raised
    # above the full pool size so nothing gets dropped.
    pools = all_single_turn_pools()
    singles = [ex for pool in pools.values() for ex in pool]
    random.shuffle(singles)
    singles = singles[:target_single]
    multis = build_multi_turn(pools, target_multi)

    blocks = [render_single(i, r) for i, r in singles] + [render_multi(pair) for pair in multis]
    random.shuffle(blocks)
    return '\n'.join(blocks)


if __name__ == '__main__':
    corpus = build()
    out_path = ROOT / 'data/instructions.txt'
    out_path.write_text(corpus, encoding='utf-8')
    print(f'Wrote {len(corpus.encode("utf-8")):,} bytes, {corpus.count("### Instruction:")} instruction blocks to {out_path}')
