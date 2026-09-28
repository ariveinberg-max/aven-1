# Catalog

`datasets/*.yaml` holds one **dataset card** per dataset, whether public or ours. Each card records provenance and license facts, and it is the source of truth for the license gate (ADR-0005, `src/neurolayer/data/catalog.py`).

```bash
uv run neurolayer catalog list                       # all datasets with license status
uv run neurolayer catalog check --purpose training   # which datasets may train shippable weights
```

## Verifying a license (WP-1.1, human task)

1. Open the dataset's license page (`license.url`, or the `source_url`).
2. Read the actual terms.
3. Update `spdx`, `commercial_use`, `derivatives`, `share_alike` and `evidence`. Quote where you read it.
4. Set `verified_by: <your name>` and `verified_on: <YYYY-MM-DD>`.
5. Open a PR. CODEOWNERS review is required for `catalog/`.

If the terms are ambiguous, leave the flags as `unclear` and ask counsel. A legal decision is recorded under `license.legal_review` with `approved_purposes`, `reviewer`, `reference` and `reviewed_on`.

## Adding a dataset

Copy the closest card, name the file `<id>.yaml` (lowercase with underscores), and fill in every field you can verify. Unknown numbers stay `null`. Do not guess.
