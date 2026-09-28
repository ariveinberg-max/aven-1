# 4. Patents and intellectual property

> **Not legal advice.** This is a technical watchlist for briefing a patent attorney. Before a public launch or a licensing deal, get a professional **freedom-to-operate (FTO)** analysis. Patent databases could not be queried directly from the research environment, so treat every entry below as a lead to verify.

## 4.1 Patent watchlist

| Holder | Patent / family | Title (as reported) | Why it matters to us |
|--------|-----------------|---------------------|----------------------|
| **Neurable** | US 11,269,414 B2; EP 3672478 B1; KR 20200101906 A; later US continuations | "Brain-computer interface with high-speed eye tracking features" | Covers hybrid gaze + brain-signal UIs. **Review before building gaze+EEG selection for AR/VR (CAP-2).** |
| **Neurable** | US 12,001,602 (issued 4 Jun 2024) | "Brain-computer interface with adaptations for high-speed, accurate, and intuitive user interactions" | Adaptive BCI user interfaces; overlaps with calibration UX |
| **Meta (CTRL-labs)** | US 10,409,371 B2; US 10,656,711 B2; ~238 EMG-related filings reported (2018–2026) | "Methods and apparatus for inferring user intent based on neuromuscular signals" | A dense thicket around EMG intent decoding. **Avoid EMG decoding without FTO.** |
| **Apple** | Filed Jan 2023, reported granted | "Biosignal sensing device using dynamic selection of electrodes" (EEG/EMG/EOG from earbuds) | Relevant if we ever build ear-EEG hardware or electrode-selection logic |
| **Arctop** | Several patents, including brain-based authentication ("Brain ID") | Cognitive-state decoding and authentication | We will not build EEG biometrics (also an ethics and legal red line, see [document 5](05-regulation-and-privacy.md)) |
| **Invasive players** (Neuralink, Synchron, Paradromics, Blackrock) | Large portfolios | Implant hardware, surgery, intracortical decoding | Low relevance to non-invasive software, but check decoder-adaptation claims before licensing into implant ecosystems |

**Prior art is on our side for the basics.** Several core methods have been published in the academic literature for years and are implemented in open-source packages:

- CSP
- Riemannian tangent-space classification
- Euclidean alignment
- Riemannian re-centering and Procrustes alignment
- EEGNet
- masked-signal pretraining

Using them as components is low-risk. Novel **combinations and product-specific methods** are where both our risk and our patent opportunities lie.

## 4.2 Our IP strategy

The company's IP rests on three assets: **proprietary models**, **proprietary data**, and the **platform**. Each is protected differently.

| Asset | Primary protection | Practices (starting now) |
|-------|--------------------|--------------------------|
| Model weights, training recipes, evaluation harness details | **Trade secret** | Private repository; weights never committed to git; access control on artifact storage |
| Novel methods (for example montage-agnostic calibration-efficient adaptation, calibration protocol and UX, online adaptation) | **Patents**, selectively | Invention-disclosure log (`docs/ip/`, created when the first invention arises). **File a provisional before any public disclosure.** |
| Datasets we collect | **Contracts + trade secret** | Consent forms that grant commercial training rights; provenance in the catalog; no raw data leaves controlled storage |
| Brand and product | Trademark | Run a trademark search on the company and product name before launch. `neurolayer` is only a code name. |
| Third-party inputs | **License compliance** | The catalog license gate (ADR-0005); a third-party model policy (ADR-0007); dependency license review in CI (planned, WP-0.9) |

## 4.3 Urgent: this repository is public

At the time of writing, `ariveinberg-max/aven-1` is a **public** GitHub repository.

- Anything pushed here is **public disclosure**. Most countries require *absolute novelty* for a patent, so a public disclosure before filing can bar the patent. The US gives a 1-year grace period; Europe and most others give none.
- Public code, model details and results **cannot be trade secrets**.

**Action (before Stage 3 work):** make the repository private (GitHub → Settings → General → Danger Zone → Change visibility). Also enable branch protection and secret scanning. The research, architecture and scaffolding in this commit are low-sensitivity. The proprietary model work that follows is not.

## 4.4 Invention-disclosure process (lightweight)

1. When an experiment produces a result that beats baselines through a *new method*, write a one-page disclosure:
   - the problem
   - prior approaches
   - what is new
   - evidence (experiment IDs)
   - the date
2. Store it in private storage, not in a public repo.
3. Decide on filing (a provisional costs little) **before**:
   - a paper
   - a talk
   - a demo to anyone outside an NDA
   - a push to any public location
