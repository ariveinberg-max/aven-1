"""Regression checker -- a real local agent that replaces manual curl-testing.

After any finetune, this sends a battery of known test prompts to Aven-1's
live /api/chat, then asks a local free model (llama3.2:3b via Ollama, no
Claude/API cost) to judge whether each response is coherent, on-topic, and
free of corruption (stray tags, truncation, unrelated splices). This is
exactly the manual work done by hand throughout tonight's session to catch
real bugs (fact-splicing, category confusion, garbled generation) -- now
automated.

This is a genuine judgment task, unlike most "roles" on the org page which
are really just scripted checks: whether a response is coherent isn't a
simple string match, it needs actual language understanding.

NOTE: this catches surface-level regressions (garbling, off-topic replies).
It does not replace the formal evaluate_capabilities.py suite or real human
preference labeling -- it's a fast first pass, not a substitute for either.
"""
import json
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAT_URL = "http://localhost:8765/api/chat"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

# Drawn directly from this session's real, documented bug history --
# each one previously caught a real regression at some point tonight.
SINGLE_TURN_CASES = [
    "who was the first president of the united states",
    "who wrote pride and prejudice",
    "who is the author of frankenstein",
    "what is the capital of spain",
    "what's the capital of brazil",
    "what day comes after monday",
    "what day comes after sunday",
    "which is bigger 7 or 3",
    "is 12 greater than 5",
    "can u help me with a math problem?",
    "I need some help",
    "how are you doing today",
    "hello",
    "thank you",
]

# The exact multi-turn sequence that originally exposed the fact-splicing bug.
MULTI_TURN_CASE = [
    "hello",
    "can u help me with a math problem?",
    "whatss 1+1",
    "whos the first president of the united states",
]


def chat(prompt, history=None):
    messages = (history or []) + [{"role": "user", "content": prompt}]
    req = urllib.request.Request(
        CHAT_URL,
        data=json.dumps({"messages": messages}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read()).get("reply", "")


def judge(prompt, response):
    # v2: the first real run of this checker found the local judge had a
    # systematic false-positive pattern -- it flagged honest declines like
    # "That's outside what I've been trained on" as BROKEN/"off-topic",
    # when that IS the correct, intended answer for a request this small
    # model genuinely can't do. Made that distinction explicit below rather
    # than trusting the model to infer it.
    judge_prompt = f"""You are checking a small language model's chat response for obvious problems. Answer ONLY with one line in this exact format, nothing else:
VERDICT: <OK or BROKEN> | REASON: <one short phrase>

Flag as BROKEN only for: garbled/repeated text, a stray tag fragment like "<|end" appearing in the visible text, or a reply that is clearly about a completely different topic than the question (e.g. answering a question about a book with an unrelated fact about a country).

Do NOT flag as BROKEN:
- factual content that might be wrong (e.g. wrong arithmetic, wrong date) -- only flag actual incoherence or off-topic replies, not incorrect facts
- an honest decline like "That's outside what I've been trained on" or "I don't have a reliable answer for that" when asked to do something -- this is a CORRECT, on-topic, intended answer for a small model admitting a real limitation, not a broken response

Question: {prompt}
Response: {response}"""
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({"model": MODEL, "prompt": judge_prompt, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        text = json.loads(r.read())["response"].strip()
    verdict = "BROKEN" if "BROKEN" in text.upper() else "OK"
    return verdict, text


def main():
    results = []

    for prompt in SINGLE_TURN_CASES:
        response = chat(prompt)
        verdict, raw = judge(prompt, response)
        results.append({"prompt": prompt, "response": response, "verdict": verdict, "judge_raw": raw})

    history = []
    for prompt in MULTI_TURN_CASE:
        response = chat(prompt, history)
        history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        verdict, raw = judge(prompt, response)
        results.append({"prompt": f"[multi-turn] {prompt}", "response": response, "verdict": verdict, "judge_raw": raw})

    broken = [r for r in results if r["verdict"] == "BROKEN"]

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [f"# Regression check -- {timestamp}", "", f"{len(results) - len(broken)}/{len(results)} passed.", ""]
    for r in results:
        mark = "❌" if r["verdict"] == "BROKEN" else "✅"
        lines.append(f"{mark} **{r['prompt']}**")
        lines.append(f"   response: {r['response']!r}")
        if r["verdict"] == "BROKEN":
            lines.append(f"   judge said: {r['judge_raw']}")
        lines.append("")

    report = "\n".join(lines)
    out_path = ROOT / "research/regression_checks" / f"{timestamp.replace(':', '')}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Saved to {out_path}")
    if broken:
        print(f"\n{len(broken)} POTENTIAL REGRESSION(S) FOUND -- review before trusting this checkpoint.")


if __name__ == "__main__":
    main()
