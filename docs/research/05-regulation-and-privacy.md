# 5. Regulation, privacy and ethics

> Not legal advice. This summarizes the landscape so we can build privacy and compliance in from day one. Engage counsel before collecting data from people or launching a product.

## 5.1 Neural data is now a regulated category

### United States: state laws

| State | Law | Effective | Scope |
|-------|-----|-----------|-------|
| Colorado | HB 24-1058 (Colorado Privacy Act amendment) | 7 Aug 2024 | First US law protecting neural data as sensitive data |
| California | SB 1223 (CCPA amendment) | 1 Jan 2025 | Neural data is "sensitive personal information" |
| Montana | SB 163 (Genetic Information Privacy Act amendment) | 1 Oct 2025 | Neural data added to genetic-privacy protections |
| Connecticut | Data Privacy Act amendment | 1 Jul 2026 | Neural data is "sensitive data"; **central nervous system only** (excludes peripheral signals such as EMG) |

In the first weeks of 2026, more bills were introduced in Alabama, California, Illinois, New York, Vermont and Virginia. Definitions of "neural data" differ between states, and some cover EMG while others do not. The patchwork will grow.

### United States: federal

- **MIND Act** (S.2925; Cantwell, Schumer, Markey; introduced 29 Sept 2025). It would direct the FTC to study neural-data collection and use and recommend standards. Status: introduced and referred to committee.
- **FDA.** The **General Wellness guidance was updated on 6 Jan 2026.** Non-invasive, low-risk products with wellness-only claims fall outside device regulation. Claims about diagnosing, treating or mitigating disease, or guiding clinical management, make the product a medical device.
  - An accessibility product for ALS patients framed as restoring communication will likely be regulated. Cognixion is running clinical trials for this reason.
  - A gaming controller is not.

  **Product claims decide the regulatory path.**

### European Union

- **AI Act Art. 5(1)(f)**, applicable since Feb 2025: **prohibits AI systems that infer emotions in the workplace and in education**, except for medical or safety reasons. Fines reach €35M or 7% of global turnover. The prohibition covers "emotion recognition from biometric data", which includes EEG-based emotion or stress inference.
- **GDPR.** Neural data processed to infer health or to identify someone is special-category data. Explicit consent is needed, along with a data protection impact assessment (DPIA).

### International

- **UNESCO Recommendation on the Ethics of Neurotechnology**, adopted 12 Nov 2025. It is non-binding but influential. It calls for:
  - strict safeguards for neural data
  - prohibiting manipulative use in recommender systems
  - restricting "nudging" and neuromarketing
  - no marketing during sleep
- **Chile** has enshrined neurorights in its constitution (2021). In 2023 its Supreme Court ordered a consumer EEG company to delete a user's brain data.

## 5.2 Engineering requirements derived from the above

These are binding requirements for our architecture. They are referenced from [security-and-privacy.md](../architecture/security-and-privacy.md).

| ID | Requirement | Where it is enforced |
|----|-------------|----------------------|
| PRIV-1 | Neural data from people (as opposed to public datasets) is classified **C3 (sensitive)** | Data classification policy |
| PRIV-2 | **Explicit opt-in consent** per purpose (product function, research, commercial model training), recorded before collection and revocable | Consent registry (Stage 8, WP-8.2) |
| PRIV-3 | **Purpose limitation:** data collected for one purpose is not used for another without new consent | Catalog `purposes` plus the license gate, extended to our own data |
| PRIV-4 | **Deletion on request**, including derived features. Model weights are retrained or covered by consent terms. | Data lineage via manifests |
| PRIV-5 | **No sale** of neural data. **No use in advertising, recommender manipulation or neuromarketing.** | Product policy |
| PRIV-6 | **No emotion or affect inference features** marketed for workplace or education use | Product policy; capability registry excludes them |
| PRIV-7 | **No biometric identification** from neural data (we must also not *enable* it: our identity-leakage audit doubles as a privacy check) | Evaluation protocol, identity probe |
| PRIV-8 | **Pseudonymous subject IDs.** Directly identifying information is stored separately from signals, with restricted access. | Data model (`subject_id` is always a pseudonym) |
| PRIV-9 | **Prefer on-device processing**; transmit decoded intents rather than raw signals by default | API design (Stage 6) |
| PRIV-10 | Encryption in transit (TLS) and at rest for C3 data; access logging | Infrastructure (Stage 6+) |
| PRIV-11 | Wellness and consumer claims only, until a deliberate decision to pursue a medical pathway | Marketing and product review |

## 5.3 Ethics position (proposed, for the founders to adopt)

- **Users own their neural data.** We are custodians.
- **Decode only what the user intends to communicate** (active and reactive control, explicit feedback). Passive inference of mental states requires separate, explicit opt-in and must never be used against the user's interests.
- Publish our evaluation methodology and honest limits (including the share of users for whom the system does not work well), and never overstate capability. The BCI field has a hype problem; honesty is a differentiator with partners.
