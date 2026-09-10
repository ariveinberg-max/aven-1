# Comparing fixed capability results

Use `python compare_capabilities.py BEFORE.json AFTER.json` to compare existing reports without loading model weights or running inference. Redirect stdout to a new file if a saved comparison is needed.

The comparator requires matching suite, tokenizer, evaluator and supporting-code hashes; prompt/decoding settings; device and runtime versions. It verifies case identities and stored exact-match scores, rejects skipped cases, and reports per-category correct counts, percentage-point changes and each improvement/regression. Different checkpoint hashes are expected. Strict source-hash matching may reject harmless code changes; rerun both checkpoints under one agreed evaluator version instead of weakening the check after seeing scores.

This command does not establish checkpoint provenance or training-data independence. It preserves each report's overlap-check label, including NOT CHECKED. In particular, an eight-word substring check cannot justify the board's current wording "zero training-data overlap confirmed": it establishes only that the scanned inputs contained no matching spans under that heuristic. Historical exposure must be checked against the actual corpus lineage. No model run, new data collection or W&B upload is performed here.

Validation: `.venv/bin/python -m unittest test_compare_capabilities test_evaluate_capabilities -v`.

Current handoff: comparison utility added separately because both evaluation runners and their documentation already had uncommitted changes from another session. Those edits were reviewed and tested but not overwritten or included in this change. Real checkpoint selection and independent evaluation remain pending explicit agreement.
