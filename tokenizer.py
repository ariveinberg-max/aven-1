"""Byte-pair encoding, trained from scratch on your own corpus.

No pretrained vocabulary is downloaded — this is the same algorithm GPT-style
models use (iteratively merge the most frequent adjacent byte pair), trained
here on whatever text you compile into the corpus. A trained tokenizer is
tied to the corpus it learned from and is saved next to the checkpoint that
uses it.
"""
import hashlib
import heapq
import json
from artifact_io import atomic_json


class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def fingerprint(self):
        """Identity includes ordered merge rules, not merely vocabulary size."""
        ordered = sorted(self.merges.items(), key=lambda item: item[1])
        payload = json.dumps([[a, b, idx] for (a, b), idx in ordered], separators=(',', ':'))
        return hashlib.sha256(payload.encode('ascii')).hexdigest()

    @staticmethod
    def _stats(ids):
        counts = {}
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
        return counts

    @staticmethod
    def _merge(ids, pair, idx):
        out = []
        i = 0
        while i < len(ids):
            if i+1 < len(ids) and ids[i] == pair[0] and ids[i+1] == pair[1]:
                out.append(idx)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, raw_bytes, vocab_size):
        if vocab_size < 256:
            raise ValueError('vocab_size must be at least 256 (the raw bytes).')
        ids = list(raw_bytes)
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        for i in range(vocab_size - 256):
            stats = self._stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            if stats[pair] < 2:
                break
            idx = 256 + i
            ids = self._merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        ids = list(text.encode('utf-8'))
        while len(ids) >= 2:
            stats = self._stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def encode_ids(self, raw_bytes):
        """Encode already-decoded bytes (e.g. a whole corpus) without a utf-8 round trip.

        O(n log n): a min-heap over adjacent mergeable pairs (keyed by merge
        rank, i.e. learned order) with a doubly linked list for O(1) splicing
        and lazy invalidation of stale heap entries, instead of one full pass
        over the whole sequence per learned merge rule (O(n * num_merges) --
        impractical once a corpus reaches tens of MB, since num_merges is
        typically ~1800 and each pass previously re-scanned everything).
        Produces identical output to the sequential full-pass approach --
        applying merges in a global priority order with correct adjacency
        tracking is the standard efficient equivalent of applying each merge
        rule as a separate full pass in the same order.
        """
        n = len(raw_bytes)
        if n < 2:
            return list(raw_bytes)
        ids = list(raw_bytes)
        nxt = list(range(1, n)) + [-1]
        prv = list(range(-1, n - 1))
        alive = [True] * n
        heap = []

        def push(i):
            j = nxt[i]
            if j == -1:
                return
            rank = self.merges.get((ids[i], ids[j]))
            if rank is not None:
                heapq.heappush(heap, (rank, i, ids[i], ids[j]))

        for i in range(n - 1):
            push(i)
        while heap:
            rank, i, a, b = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j == -1 or ids[i] != a or ids[j] != b:
                continue  # stale entry: this position's pair already changed
            ids[i] = rank
            k = nxt[j]
            nxt[i] = k
            if k != -1:
                prv[k] = i
            alive[j] = False
            p = prv[i]
            if p != -1:
                push(p)
            push(i)
        out = []
        i = 0
        while i != -1:
            out.append(ids[i])
            i = nxt[i]
        return out

    def decode(self, ids):
        return b''.join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')

    def save(self, path):
        ordered = sorted(self.merges.items(), key=lambda kv: kv[1])
        atomic_json(path, {'merges': [[a, b] for (a, b), _ in ordered]})

    def load(self, path):
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or not isinstance(data.get('merges'), list) or len(data['merges']) > 16384 - 256:
            raise ValueError('Invalid tokenizer merge table.')
        vocab = {i: bytes([i]) for i in range(256)}
        merges = {}
        idx = 256
        for pair in data['merges']:
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(type(value) is not int or not 0 <= value < idx for value in pair)
                    or tuple(pair) in merges):
                raise ValueError(f'Invalid tokenizer merge at token {idx}.')
            a, b = pair
            merges[(a, b)] = idx
            vocab[idx] = vocab[a] + vocab[b]
            idx += 1
        self.merges = merges
        self.vocab = vocab
        return self
