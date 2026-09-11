# Aven capability-v1

Scope: issue #2's fixed capability tests only. No training-data expansion, assistant tools, training, or generated-code execution. This is a small diagnostic suite, not a validated intelligence benchmark. No real Aven checkpoint has been evaluated in this change.

The frozen collection contains 30 original items, six each for comprehension, instruction following, arithmetic, basic facts and Python code reading. Provenance is recorded per item. Code reading tests predict output; they do not measure general programming ability. Facts are elementary probes, not evidence of broad factual reliability. Do not train on these prompts or answers. Future edits require a new suite version; retain v1 for comparisons.

Protocol: greedy decoding, fixed plain or instruction prompt format, 32 generated-token limit, stop at first newline or end marker, whole-response exact match with edge whitespace stripped. Case matters. A semantically correct response with extra explanation fails this strict instruction-following protocol; raw responses are retained for manual error analysis. Report numerator/denominator by category, not a single intelligence score. Prompts exceeding the checkpoint's context are explicitly marked, counted as failures and excluded from the evaluated count; never silently truncated. With only six items per category, each item changes the score by 16.7 percentage points. Do not tune decoding after viewing scores.

## Reproducible command

Use an explicitly agreed checkpoint and hashes; the runner has no default checkpoint path. The placeholder paths below must be replaced before running:

```bash
.venv/bin/python evaluate_capabilities.py \
  --checkpoint /absolute/path/latest.pt \
  --tokenizer /absolute/path/tokenizer.json \
  --expect-sha256 CHECKPOINT_SHA256 \
  --expect-tokenizer-sha256 TOKENIZER_SHA256 \
  --corpus /absolute/path/training-corpus.txt \
  --format plain --device cpu --max-new-tokens 32 \
  --output /absolute/path/new-evaluation-report.json
```

The runner reads model artifacts and writes only a new report. It does not create optimizers, modify checkpoints, or log to W&B. Reports capture suite/evaluator/checkpoint/tokenizer hashes, code revision, architecture, step, runtime versions, settings, raw outputs and category scores. Use identical tokenizer hash, suite hash, prompt format, device/runtime and decoding settings for longitudinal comparisons. A tokenizer vocabulary-size match alone is insufficient proof of tokenizer identity. Keep result reports outside training inputs.

## Overlap and limitations

Current train.py fits BPE on the entire corpus before making a contiguous 90/10 token split. The validation text therefore influences tokenizer construction, and a contiguous split is not an independent document-level test set. Repeated content across the split and books is another overlap risk.

Supply all available pretraining and fine-tuning corpora via repeated --corpus arguments. The runner rejects normalized exact eight-word prompt spans found in those inputs and rejects duplicate suite IDs/prompts. Omission is recorded as NOT CHECKED, never as clean. Short questions, shared facts, paraphrases and semantic overlap can escape this heuristic; passing is not proof of non-contamination. Original authorship does not guarantee unseen concepts. Do not publish claims of independent performance without reviewing overlap findings and training provenance.

A genuinely separate held-out text collection and token loss/perplexity evaluation remain future issue #2 work to agree with Ari/Claude. These short capability questions do not substitute for that collection. A later-checkpoint comparison and W&B publication also remain outstanding; no issue completion or checkpoint recovery is implied.

## Validation

Run `.venv/bin/python -m unittest test_evaluate_capabilities -v`. Tests use a freshly initialized tiny model in a temporary directory, verify all 30 cases are processed, check exact-match and overlap behavior, and confirm checkpoint/tokenizer hashes remain unchanged and wrong identity aborts without a report. Tiny-model outputs are harness validation, not Aven research results.

## Streaming evaluation update

The overlap scan now normalizes fixed-size text chunks with boundary carry, keeping
the original eight-word substring protocol and prompt-span ordering without reading
entire training corpora into memory. Model loading pins the same open checkpoint
file that was hashed, protecting report identity during atomic saves. Reports also
record hashes of the model, tokenizer and artifact-loading source files.
External-text evaluation is available through `eval_heldout.py`; its `external-text-v2`
report records bounded-chunk tokenization. Dataset folder exclusion alone does not
prove a historical checkpoint was never trained on that text.
