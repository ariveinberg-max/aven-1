# ADR-0007: Third-party foundation-model policy

- **Status:** Accepted
- **Date:** 2026-09-26

## Context
Open EEG foundation models are useful baselines and possible initializations, but:

- their weights inherit terms from their pretraining data (often clinical data under data-use agreements)
- some use custom restrictive licenses (REVE: no redistribution, revocable access)
- checkpoints may be pickles that can execute code when loaded
- 2026 audits show identity and dataset leakage in their representations

## Decision
1. Third-party models may be used as **baselines** (benchmark purpose) when their code license allows it. Record the model, version, weight hash and license in the run manifest.
2. Third-party weights may be used as an **initialization for shippable models** only after a legal review of both the weight license and the pretraining-data terms. The review is recorded as a model card in `catalog/models/` (created with the first such model).
3. Load weights only via `safetensors` or `torch.load(..., weights_only=True)`. Pin by SHA-256.
4. Every foundation-model result is reported alongside identity and dataset-ID probe results.

## Consequences
Our proprietary model must be pretrained on license-clean data, which is slower. That clean provenance is the business moat.

## Alternatives considered
- "Everyone uses LaBraM weights": acceptable for papers, not for licensing deals.
