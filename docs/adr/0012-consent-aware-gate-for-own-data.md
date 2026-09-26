# ADR-0012: Consent-aware usage gate for our own recordings

- **Status:** Proposed (needs founder and counsel sign-off before pilot data is used for training)
- **Date:** 2026-09-26

## Context
The dataset license gate (ADR-0005) allows every purpose for **internal** cards (`access: internal`, `LicenseRef-Internal*`). That is right for synthetic data. It is wrong for data we record from people. Consent there is per participant and per purpose (product, research, commercial training; PRIV-2). One pilot dataset will mix participants who allowed commercial training and participants who did not. A card-level gate cannot express that.

Stage 8 tooling already records consent in an append-only ledger (`neurolayer.data.consent.ConsentRegistry`). Recording refuses participants without `research` consent. `ConsentRegistry.participants(purpose)` lists who may be used for a purpose.

## Proposed decision
1. Cards for our own recordings use `license.spdx: LicenseRef-Consent`, **not** `LicenseRef-Internal`. The card-level gate then treats them as not internal, and a new rule applies:
   - `exploration` and `benchmark` require `research` consent;
   - `training` requires `commercial_training` consent.
2. The pilot loader filters participants with `ConsentRegistry.participants(purpose)` **at load time**, for the run's purpose. The run manifest records the ledger's SHA-256 and the included pseudonyms, so a run can be audited and revocations traced.
3. A revocation (PRIV-4) triggers deletion of the raw recordings, and exclusion from future runs. Models already trained on the data are listed in a revocation log. Whether they must be retrained is a counsel question, recorded here once answered.
4. `catalog/templates/neurolayer_pilot.yaml` becomes a real card only after this ADR is accepted.

## Consequences
- Changing the gate touches `neurolayer.data.catalog`, which ADR-0005 governs, so it waits for this ADR.
- Training data for commercial models can only ever contain people who agreed to it, and that is checkable after the fact.
- Until acceptance, pilot recordings may be collected (with `research` consent) but not used for training.
