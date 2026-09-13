"""One-off test: how often does a local AI judge agree with Ari's real
preference labels, on the exact same (prompt, response_a, response_b) pairs?

Not a permanent pipeline component -- a direct, honest measurement to answer
a real question: would switching to RLAIF actually track what Ari wants?
"""
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import preferences

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"


def ai_judge(prompt, response_a, response_b):
    judge_prompt = f"""You are comparing two AI responses to the same prompt and deciding which is better. Consider correctness, tone, and how well it actually answers the prompt. Answer ONLY with one line in this exact format, nothing else:
WINNER: A or WINNER: B

Prompt: {prompt}
Response A: {response_a}
Response B: {response_b}"""
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({"model": MODEL, "prompt": judge_prompt, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        text = json.loads(r.read())["response"].strip().upper()
    if "WINNER: A" in text or text.strip() == "A":
        return "a"
    if "WINNER: B" in text or text.strip() == "B":
        return "b"
    return None  # judge didn't give a clear answer


def main():
    random.seed(7)
    decided = preferences.all_decided()
    sample = random.sample(decided, min(30, len(decided)))
    agree = disagree = unclear = 0
    for row in sample:
        ai_pick = ai_judge(row['prompt'], row['response_a'], row['response_b'])
        human_pick = row['winner']
        if ai_pick is None:
            unclear += 1
            mark = '?'
        elif ai_pick == human_pick:
            agree += 1
            mark = '='
        else:
            disagree += 1
            mark = 'X'
        print(f"[{mark}] human={human_pick} ai={ai_pick}  {row['prompt'][:50]!r}")
    total = agree + disagree
    print(f"\nAgreement (excluding unclear): {agree}/{total} = {agree/total:.1%}" if total else "No clear comparisons.")
    print(f"Unclear AI responses: {unclear}/{len(sample)}")


if __name__ == '__main__':
    main()
