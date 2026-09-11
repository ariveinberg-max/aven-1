"""Explicit, source-backed note recall. Does not modify notes or model weights."""
import math
import re
from collections import Counter

_WORDS = re.compile(r"\w+", re.UNICODE)
_STOP = set('a an the is are was were what which who where when how do does did my me i you your about of to in on for and or please notes note memory memories saved remember recall find search'.split())


def terms(text):
    return [word for word in _WORDS.findall(text.casefold()) if word not in _STOP]


def search_records(query, records, limit=3):
    """BM25 ranking with title weighting; return excerpts with inspectable IDs."""
    query_terms = set(terms(query))
    if not query_terms or not records:
        return []
    docs = [Counter(terms(r['title']) * 2 + terms(r['content'])) for r in records]
    lengths = [sum(d.values()) for d in docs]
    average = sum(lengths) / len(docs) or 1
    frequency = {word: sum(word in doc for doc in docs) for word in query_terms}
    ranked = []
    for record, doc, length in zip(records, docs, lengths):
        score = 0.0
        for word in query_terms:
            count = doc[word]
            if count:
                inverse = math.log(1 + (len(docs) - frequency[word] + .5) / (frequency[word] + .5))
                score += inverse * count * 2.2 / (count + 1.2 * (.25 + .75 * length / average))
        if score:
            content = record['content']
            matches = [m for m in _WORDS.finditer(content) if m.group().casefold() in query_terms]
            start = max(0, matches[0].start() - 120) if matches else 0
            excerpt = ('…' if start else '') + content[start:start + 700]
            if start + 700 < len(content):
                excerpt += '…'
            ranked.append(dict(id=record['id'], title=record['title'], excerpt=excerpt, score=round(score, 6)))
    return sorted(ranked, key=lambda r: (-r['score'], r['id']))[:max(0, min(limit, 10))]


def recall_query(message):
    if not isinstance(message, str):
        return None
    match = re.fullmatch(r'\s*(?:/recall\b|(?:search|find)\s+(?:my\s+)?(?:notes|memories)\b|(?:what\s+do\s+you\s+)?remember\s+about\b)\s*:?[ \t]*(.*?)\s*', message, re.I | re.S)
    return match.group(1) if match else None


def answer_recall(message, records):
    query = recall_query(message)
    if query is None:
        return None
    sources = search_records(query, records)
    if not sources:
        reply = 'No matching saved notes found.' if terms(query) else 'Add a topic after /recall to search your saved notes.'
    else:
        reply = 'Matching saved notes:\n\n' + '\n\n'.join(
            f"[{r['id']}] {r['title'] or 'Untitled'}\n{r['excerpt']}" for r in sources)
    return dict(reply=reply, activity=[], tool='memory', sources=sources)
