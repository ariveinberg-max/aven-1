# EXP-YYYYMMDD-<slug>

- **Status:** planned | running | done
- **Owner:**
- **Work package / hypothesis:** WP-x.y / H#
- **Config:** `configs/experiments/<name>.yaml` (config hash: `<fill after run>`)
- **Official run id(s):** `<artifacts/runs/...>`

## Question
What do we want to learn? One sentence.

## Hypothesis and prediction (write BEFORE running)
- Hypothesis:
- Predicted outcome, with numbers (for example "BA@10 on R2 improves by ≥ 3 pp over B3"):
- What would falsify it:

## Setup
Datasets (catalog ids and license-gate purpose), regime(s), budgets, decoder(s), seeds, and any deviation from the standard protocol.

## Results
Table or plot of BA@k, UUR@k, AUCEC and TTC, with CIs. Link `summary.json`.

## Leakage controls
Shuffle control, identity probe and dataset-ID probe results (required for gate evidence).

## Conclusion
Supported / refuted / inconclusive. What we will do next. Negative results are valuable. Record them.
