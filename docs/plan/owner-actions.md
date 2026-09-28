# Owner actions: everything that needs a human

**Date:** 2026-09-28 · The agent did everything around these items; each one below needs a person, an account setting, money or a signature. Ordered by urgency.

## 1. Make the repository private (5 minutes, urgent)

The proprietary model code is in a public repository. There is no tool the agent can use to change repository visibility.

1. Open https://github.com/ariveinberg-max/aven-1/settings
2. Scroll to **Danger Zone** → **Change repository visibility** → **Make private** → confirm.
3. Same page, **Branches** → add a rule for `main`: require a pull request, require status checks (the CI jobs), no force pushes.
4. https://github.com/settings/security → enable two-factor authentication if it is not on.

Claude cloud sessions keep working on a private repository as long as the Claude GitHub App is installed on it.

## 2. Verify dataset licenses (WP-1.1; about 2 minutes per dataset)

AGENTS.md rule 5 makes this a human decision. The agent could not open the primary pages: physionet.org, gigadb.org and bnci-horizon-2020.eu are blocked from its environment. Below is what it could find. Open each link, read the license line, and tell the agent what you saw. It will record `verified_by`/`verified_on` with your name.

| Dataset | What to open | Card says | Evidence the agent found | What to confirm |
|---------|--------------|-----------|--------------------------|-----------------|
| `physionet_mi` (downloaded, 109 people) | https://physionet.org/content/eegmmidb/1.0.0/ (see "License (for files)") | ODC-By 1.0 | Search engines index the page "License for EEG Motor Movement/Imagery Dataset v1.0.0" as the Open Data Commons Attribution License v1.0. **Conflict:** the AWS Open Data registry says PhysioNet's *legacy* `physionet-pds` bucket is under ODC-PDDL (public domain). Both allow commercial use and derivatives | The exact license name on the page. ODC-By needs attribution; PDDL does not |
| `cho2017` | https://gigadb.org/dataset/100295 | Unknown | GigaDB datasets are usually CC0; not confirmed | License line on the dataset page |
| `lee2019_mi` | https://gigadb.org/dataset/100542 | Unknown | Same as above | License line on the dataset page |
| `bnci2014_001` | https://bnci-horizon-2020.eu/database/data-sets (dataset 001-2014) | CC BY-ND 4.0 | Search results describe 001-2014 as CC BY-ND 4.0, licensor Graz University of Technology | CC BY-ND 4.0; "no derivatives" keeps it benchmark-only |

## 3. Give cloud sessions access to the other dataset hosts

PhysioNet works through its AWS mirror (`scripts/fetch_physionet_mirror.py`). Every other development dataset needs its host allowed. In the Claude app: open this session's cloud environment menu (session title bar) → **Edit** → **Network access** → **Custom**. Tick "Also include default list of common package managers" and paste:

```text
physionet.org
*.physionet.org
gigadb.org
*.gigadb.org
ftp.cngb.org
bnci-horizon-2020.eu
*.bbci.de
zenodo.org
figshare.com
*.figshare.com
openneuro.org
huggingface.co
*.huggingface.co
```

Alternatively, run `make setup-ml` and `uv run neurolayer data fetch <dataset>` on the Windows GPU PC.

## 4. Hardware for dogfooding and the pilot (money)

The final device choice waits for the real R3 results ([device memo](../templates/device-selection-memo.md)). One of each lets us test the bridge on real heads now.

| Item | Why | Price |
|------|-----|-------|
| Neurosity Crown | 8 channels (CP3, C3, F5, PO3, PO4, F6, C4, CP4) at 256 Hz; LSL and BrainFlow. Neurosity's own guides list these specs | $1,499 per Neurosity's site (checked 2026-09-28 via search; confirm at checkout) |
| OpenBCI Cyton (8-ch) or Cyton + Daisy (16-ch), with a gel electrode cap in a motor layout (C3/Cz/C4/FC3/FC4/CP3/CP4…) | Research-grade reference; lets us place electrodes where CAP-1 needs them | `TODO(verify)`: shop.openbci.com was not reachable for the agent |
| Light sensor (photodiode) into a spare analog input | Cue-timing test before the pilot ([protocol §5](../guides/pilot-protocol.md)) | Small |

Before buying the Crown, check its data terms: whether we may store raw data and train commercial models on it (a counsel question, below).

## 5. Counsel brief (bring these documents and questions)

Documents: [consent form draft](../templates/consent-form.md), [pilot protocol](../guides/pilot-protocol.md), [ADR-0012](../adr/0012-consent-aware-gate-for-own-data.md), [regulation summary](../research/05-regulation-and-privacy.md).

Questions:

1. Which jurisdictions' neural-data laws apply to where we recruit? Colorado, California, Montana and Connecticut already treat neural data as sensitive, and more states have bills pending. What must the consent form say for each, and for GDPR if any EU participant?
2. Is purpose-separated consent (product / research / commercial training) sufficient, and how must withdrawal work? In particular: must models already trained on a withdrawn participant's data be retrained?
3. Retention period for raw recordings, and deletion timelines.
4. Do we need an independent ethics (IRB-style) review for the pilot?
5. Vendor terms: may we store and commercially use data recorded with the Neurosity Crown or OpenBCI hardware?
6. Product claims: confirm that "calibration-efficient intent decoding for gaming and computer control" stays under the FDA General Wellness guidance, and which accessibility claims would not.
7. EU AI Act Art. 5(1)(f): confirm that our products never infer emotions in workplace or education settings, and how to state that in terms of use.
8. IP: timing of provisional patent filings relative to any public disclosure (the repository was public until step 1).
