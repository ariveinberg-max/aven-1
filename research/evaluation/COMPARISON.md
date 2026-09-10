# Comparing fixed capability results

Use `python compare_capabilities.py BEFORE.json AFTER.json` to compare existing reports without loading model weights or running inference. Redirect stdout to a new file if a saved comparison is needed.

The comparator requires matching suite, tokenizer, evaluator and supporting-code hashes; prompt/decoding settings; device and runtime versions. It verifies case identities and stored exact-match scores, rejects skipped cases, and reports per-category correct counts, percentage-point changes and each improvement/regression. Different checkpoint hashes are expected. Strict source-hash matching may reject harmless code changes; rerun both checkpoints under one agreed evaluator version instead of weakening the check after seeing scores.

This command does not establish checkpoint provenance or training-data independence. It preserves each report's overlap-check label, including NOT CHECKED. In particular, an eight-word substring check cannot justify the board's current wording "zero training-data overlap confirmed": it establishes only that the scanned inputs contained no matching spans under that heuristic. Historical exposure must be checked against the actual corpus lineage. No model run, new data collection or W&B upload is performed here.

Validation: `.venv/bin/python -m unittest test_compare_capabilities test_evaluate_capabilities -v`.

Current handoff: comparison utility added separately because both evaluation runners and their documentation already had uncommitted changes from another session. Those edits were reviewed and tested but not overwritten or included in this change. Real checkpoint selection and independent evaluation remain pending explicit agreement.

## Regression checks and malformed reports

Use `python compare_capabilities.py BEFORE.json AFTER.json --fail-on-regression` for an automated check. Exit 0 means the comparison passed with no newly incorrect cases; exit 1 means at least one previously correct case regressed; exit 2 means invalid/incompatible inputs or unreadable files. This checks individual cases even when the aggregate score stays unchanged. JSON output includes regression/improvement counts and a same-checkpoint indicator.

Reports must contain valid SHA-256 identities, supporting-code hashes, a positive integer token budget, nonempty case IDs and prompts, positive integer prompt lengths and lists of accepted answer strings. In particular, a string cannot substitute for an answer list: that would turn membership into substring matching and permit false positives. Missing provenance is not silently treated as matching provenance. These are report-consistency checks, not cryptographic attestation of the original run.

## Error analysis without changing scores

`python capability_errors.py REPORT.json --output NEW-WORKSHEET.md` creates an exclusive new Markdown worksheet with checkpoint/suite/tokenizer identity, recomputed category counts, and all failed or skipped cases. It distinguishes empty responses, case-only mismatches and prompts exceeding context; all other incorrect answers require human review. No causal diagnosis or semantic rescoring is invented. Model text is escaped in the worksheet. Keep the frozen questions out of training data, and use failures to design separate development experiments rather than teaching the answers to this test.
