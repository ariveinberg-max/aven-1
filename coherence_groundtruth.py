"""Ground-truth (prompt, response, label) set for validating any coherence
checker -- built 2026-09-15 after FOUR separate generation-fix attempts
(dictionary filter, adaptive temperature, LLM judge, decaying temperature)
all failed for what looks like the same underlying reason: no attempt had
a reliable way to MEASURE whether a response was actually garbled. Every
example here was manually inspected and labeled by a human (not the
checker being tested) during this project's real investigations on
2026-09-14/15 -- see research/TASKS.md for where each came from.

Use this to compute real precision/recall for any coherence-check
candidate before trusting a handful of eyeballed examples again.
"""

# label: True = coherent/good, False = garbled or genuinely off-topic
GROUND_TRUTH = [
    ("Introduce yourself", "I am a small language model built and trained from random weights on this computer.", True),
    ("Can you help me with a science project?", "That's outside what I've been trained on. I can help with the kinds of things I'm not sure — I've been trained on.", False),
    ("Is it better to work alone or with a team?", "Working alone gives you more control and fewer meetings, but a good team gets you further on anything complicated, since no one bring obson has every skill a hard proble.", False),
    ("What is more important, talent or hard work?", "Talent is overrated compared to how it gets talked about. Most people who look naturally gifted actually put the problems, like p an actually put in enfa, as where she expensible.", False),
    ("Do you think technology makes life better or worse?", "Mostly better, on balance : s to find out what you don't experience feelings, but everything is running fine.", False),
    ("Should people always tell the truth, even if it hurts?", "I am a small language model built and trained from random weights on this computer.", False),  # identity reply, wrong topic
    ("Hey there", "Hey! How can I help you?", True),
    ("Thank you", "Anytime!", True),
    ("Would you rather be invisible or be able to fly?", "I'd pick flying. Invisibility is mostly useful for avoiding people, but flying would actually let you get somewhere and see the world differently.", True),
    ("What makes a good friend?", "Reliability is the real test: does this person show up on a bad day, not just a good one? Everything else, like shared interests, matters much less than that.", True),
    ("Who was the first president of the United States?", "George Washington was the first president of the United States.", True),
    ("What is the capital of France?", "The capital of France is Paris.", True),
    ("Is it better to be cautious or take risks?", "Neither one is right on its own. Being too cautious means missing real opportunities, while being too reckless means the failures pile up faster than you can recover from them.", True),
    ("Could you assist me?", "3 times 3 is 9.", False),
    ("I need some help", "Sofives you more control and fewer meetings, but a", False),
    ("Could you assist me?", "Sure — what is eter than 1.", False),
    ("What is more important, talent or hard work?", "Teakes at all.", False),
    ("Do you think technology makes life better or worse?", "woperations. An hour a day for a Mondon't know tha", False),
    ("Hello!", "Hi there! is generally considered the longest rive", False),
    ("Who are you?", "I don't have feelings, but I'm working fare wrote ", False),
    ("I need some help", "Hi! What seemther else, need oxygen to survive.", False),
    ("What is the best way to learn something new?", "Bentuish esself. An orchid", False),
    ("Thank you", "orange, graht", False),
    ("Who wrote Pride and Prejudice?", "Pride and Prejudice was written by Jane Austen.", True),
    ("Who wrote Pride and Prejudice?", "Priysday to Jane Austen.", False),
    ("Who is the author of Frankenstein?", "Frankenstein was written by Mary Shelley.", True),
    ("Who is the author of Frankenstein?", "The Adventures of Sherlock Holmes was written by ArConan Doyle.", False),
    ("I need some help", "Sure — what do you need help with?", True),
    ("I need some help", "I'm listening. Invisibility, honestly — it seems more useful day to day, like avoiding an awkward conversation or watching something unfold without being noticed.", False),
    ("whatss 1+1", "1 plus 1 is 1.", False),  # known arithmetic limitation, still incoherent output for this purpose
]
