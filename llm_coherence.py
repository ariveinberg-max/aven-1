"""WIRED IN as of 2026-09-15 (server.py's /api/preferences/pair retry loop).
The initial 2026-09-14 hand-eyeballed test (6 examples) suggested this
caught only 1/6 real defects -- but that sample was far too small to trust,
per this project's own repeated lesson about single-run/small-sample
comparisons. Properly measured on 2026-09-15 against a real 30-example
ground-truth set (coherence_groundtruth.py, built from a week of manually-
verified examples): 83.33% accuracy, 73.68% recall on real garbled
examples, 0% false positives on genuinely good responses -- clearly better
than both the dictionary-based coherence.py attempt (53.33% accuracy) and
the production plausible() check alone (43.33% accuracy, only 10.53%
garbage recall). See research/TASKS.md's 2026-09-15 entry for the full
comparison and reasoning.

LLM-judge coherence check for generated response PAIRS, for use in
server.py's /api/preferences/pair retry loop and raft.py's candidate
generation -- replacing the cheap plausible() heuristic (non-alphabetic
start / stray end-marker only), which live-testing on 2026-09-14 showed
misses most real defects (garbled word fragments diluted across an
otherwise-normal sentence, topic-bled canned responses).

Reuses the exact judge pattern already proven in agents/regression_checker.py
(local Ollama, llama3.2:3b, no cost) rather than inventing a new mechanism --
a dictionary-heuristic attempt (see coherence.py) was tried first and failed
to catch real defects; this judges coherence directly instead of trying to
hand-engineer a detector for every way text can go wrong.

Unlike regression_checker.py's judge() (which only checks ONE response
against a prompt, post-hoc), this checks a PAIR at once and is meant to run
inside the generation retry loop -- so it needs to be fast enough not to
make pair-generation unusably slow. Uses a short timeout and treats a judge
failure (Ollama down, timeout) as "assume coherent" so a broken judge never
blocks the whole labeling pipeline -- this is a filter to catch obvious
defects, not a hard gate the system depends on.
"""
import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"


def judge_coherent(prompt, response, timeout=15):
    """Returns True if the judge says this response is coherent and on-topic
    for this prompt, False if it flags real garbling/off-topic content.
    Returns True (fail open) on any judge error -- see module docstring."""
    judge_prompt = f"""You are checking one response from a small language model for real defects. Answer ONLY with one word: OK or BROKEN.

Flag as BROKEN only for: garbled or invented word fragments (not real words), a stray tag fragment like "<|end" in the visible text, duplicated/repeated text, or a reply that answers a completely different question than the one asked.

Do NOT flag as BROKEN:
- factual content that might be wrong (e.g. wrong arithmetic, wrong date) -- only flag actual incoherence or off-topic replies, not incorrect facts
- an honest decline like "That's outside what I've been trained on" when asked to do something -- this is a correct, on-topic answer for a small model admitting a real limitation

Question: {prompt}
Response: {response}"""
    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps({"model": MODEL, "prompt": judge_prompt, "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = json.loads(r.read())["response"].strip().upper()
        return "BROKEN" not in text
    except Exception:
        return True  # fail open -- a down/slow judge should never block labeling
