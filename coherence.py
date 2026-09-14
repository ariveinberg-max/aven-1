"""EXPERIMENTAL, NOT WIRED IN -- 2026-09-14 attempt at a dictionary-based
coherence filter, kept for the record rather than deleted. Live-tested
against real garbled and clean examples from this session's KTO/PPO/RAFT
output: caught only 1/6 real defects after fixing two bugs (missing
plurals/contractions in the word list, an overly-short repeated-substring
window) -- a ratio-based check dilutes 1-2 fake words into insignificance
across an otherwise-normal-length sentence, which is the actual common
failure pattern observed live, not full-sentence gibberish. Also produced
at least one false positive on genuinely clean text ("Anytime!", a common
word missing from this specific dictionary file).
Conclusion: this heuristic approach is not reliable enough to use -- do
NOT import this into server.py or raft.py's generation paths as-is. The
real next attempt should follow this project's already-proven LLM-judge
pattern (see agents/regression_checker.py, agents/rlaif_agreement_test.py)
instead of a from-scratch dictionary heuristic. See research/TASKS.md's
2026-09-14 entry for the full writeup.

Original design notes below, for context on what was attempted:

Built 2026-09-14 after live-testing found that heuristic missed real,
observed defects: mid-sentence word-fragment garbling ("no one bring obson
has every skill a hard proble") and duplicated trailing substrings
("...I've practiced.ve practiced."), both from live KTO/PPO/RAFT output on
this project's actual checkpoints. Two checks, in addition to the original
heuristic:
  1. Real-word ratio against a real dictionary (catches invented/garbled
     word fragments a plain alphabetic-start check can't see).
  2. Repeated-substring detection (catches duplicated trailing text).

Mac-only dependency (/usr/share/dict/words) -- this module is only used by
server.py (loopback-only local panel) and raft.py, both run on the Mac in
this project's setup, never the Windows training pipeline.
"""
import re
from pathlib import Path

_DICT_PATH = Path('/usr/share/dict/words')
_WORDS = None


def _load_words():
    global _WORDS
    if _WORDS is None:
        try:
            _WORDS = set(w.strip().lower() for w in _DICT_PATH.read_text(errors='ignore').splitlines() if w.strip())
        except OSError:
            _WORDS = set()
    return _WORDS


def _has_repeated_substring(text, min_len=15):
    """Catches duplicated trailing fragments like '...practiced.ve practiced.'
    -- a substring of at least min_len real (non-space) characters that
    occurs more than once in the text."""
    compact = text
    n = len(compact)
    seen = set()
    for i in range(n - min_len + 1):
        chunk = compact[i:i + min_len]
        if chunk.strip() != chunk or len(chunk.strip()) < min_len:
            continue  # skip chunks touching whitespace padding, not a real repeat
        if chunk in seen:
            return True
        seen.add(chunk)
    return False


def is_coherent(text, min_real_word_ratio=0.65):
    """Real coherence check for a generated response. Returns False for
    empty/too-short text, non-alphabetic starts, stray end-markers (the
    original plausible() checks), AND for garbled word fragments or
    duplicated trailing substrings (new checks)."""
    if not text or len(text) < 3 or not text[0].isalpha() or '<|' in text:
        return False
    words = _load_words()
    if words:
        tokens = re.findall(r"[A-Za-z']+", text)
        if tokens:
            real = sum(1 for t in tokens if len(t) < 3 or t.lower().strip("'") in words)
            if real / len(tokens) < min_real_word_ratio:
                return False
    if _has_repeated_substring(text):
        return False
    return True
