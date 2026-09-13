"""RLAIF auto-labeler -- a real, scoped agent, not a full RLHF replacement.

Only ever touches AI_LABEL_PROMPTS (server.py) -- the substantive-prompt
bucket where a direct measurement (agents/rlaif_agreement_test.py) found the
local AI judge agrees with Ari's real labels ~100% of the time, versus 53.3%
overall. The short small-talk bucket (HUMAN_LABEL_PROMPTS) is deliberately
never auto-labeled; that measurement found the AI judge barely better than
chance there.

Every row this writes is tagged source='ai' in preferences.db -- it is never
silently blended with real human labels. Run manually or on a schedule.
"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import preferences
from server import AI_LABEL_PROMPTS

SERVER_URL = "http://localhost:8765/api/preferences/pair"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"


def generate_pair(prompt):
    req = urllib.request.Request(
        SERVER_URL,
        data=json.dumps({"prompt": prompt}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


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
    return None


def main():
    labeled = skipped = 0
    for prompt in AI_LABEL_PROMPTS:
        try:
            pair = generate_pair(prompt)
        except Exception as e:
            print(f"[skip] {prompt!r}: could not generate pair ({e})")
            skipped += 1
            continue
        if pair['response_a'].strip() == pair['response_b'].strip():
            print(f"[skip] {prompt!r}: identical responses, nothing to judge")
            skipped += 1
            continue
        winner = ai_judge(pair['prompt'], pair['response_a'], pair['response_b'])
        if winner is None:
            print(f"[skip] {prompt!r}: judge gave an unclear verdict")
            skipped += 1
            continue
        preferences.add(pair['prompt'], pair['response_a'], pair['response_b'], winner, source='ai')
        print(f"[ai={winner}] {prompt!r}")
        labeled += 1
    print(f"\nLabeled {labeled}, skipped {skipped}. Real totals: {preferences.count_by_source()}")


if __name__ == '__main__':
    main()
