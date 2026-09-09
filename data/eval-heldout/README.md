# Genuinely held-out evaluation set

These two books are **never** compiled into `data/training.txt` by
`sources.compile_corpus()` because they live outside `data/sources/`. A
checkpoint trained on this corpus has seen zero tokens of either file, at
any point (pretraining or fine-tuning).

This is a different thing from `train.py`'s internal 90/10 split. That split
takes the *last 10% of the same encoded corpus* as "held-out" -- same books,
same authors, same era, just a different slice of the same training.txt. A
model can score well there just by having memorized the stylistic and
lexical patterns of the 57 other books it trained on; it says little about
generalization to unseen text. See "The corpus was badly undersized..."
follow-up in `WRITEUP.md` for the full discussion.

## What's here and why

- `sign-of-four.txt` (Arthur Conan Doyle, *The Sign of the Four*, 241,390
  bytes) -- an author/genre **already well represented** in training
  (`sherlock-holmes.txt`, `study-in-scarlet.txt`, and
  `hound-of-baskervilles.txt` remain in `data/sources/`). This tests whether
  the model generalizes within a style it has seen a lot of, to a specific
  book it hasn't -- the harder, more informative test.
- `moby-dick.txt` (Herman Melville, 1,256,435 bytes) -- Melville does not
  appear anywhere else in the corpus. This tests generalization to an author
  and voice entirely absent from training -- a different, easier-to-interpret
  signal (a bad score here isn't surprising; a good score would be a real
  finding).

Both together: 1,497,825 bytes, about 3.7% of the pre-holdout corpus
(39,987,621 -> 38,516,269 bytes after recompiling).

## How to use it

Run `eval_heldout.py` (repo root) against a checkpoint:

```
python3 eval_heldout.py checkpoints/latest.pt
```

It loads the checkpoint's own `tokenizer.json`, encodes both files, and
reports loss/perplexity per file plus a combined figure, using a full
non-overlapping pass over each held-out text (not train.py's random-batch
sampling) so the number is reproducible run to run for the same checkpoint.

This does not replace `evaluate_capabilities.py` -- that measures capability
(can the model do tasks). This measures a genuine held-out language-modeling
loss/perplexity: how well the model predicts text it has never trained on at
all, as opposed to a different slice of text it has.

## If you add or change what's held out

Keep it small relative to the full corpus (a few percent, not more -- the
model still needs the data) and keep this README in sync with whatever
files are here. Never move a file from here back into `data/sources/` and
recompile without noting it in `WRITEUP.md` -- doing so silently would make
past held-out numbers for these files incomparable to future ones.
