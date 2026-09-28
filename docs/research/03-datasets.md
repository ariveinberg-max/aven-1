# 3. Public datasets and licensing

The dataset's license decides what we can do with it. We track three levels of use:

| Purpose | Meaning | Typical requirement |
|---------|---------|---------------------|
| `exploration` | Look at the data locally and develop pipeline code. Nothing trained on it ships. | License does not prohibit commercial use |
| `benchmark` | Report evaluation numbers on it (internally or publicly) | License **verified by a human** and not non-commercial |
| `training` | Train weights that may ship in a product or be licensed to a partner | Verified, commercial use **yes**, derivatives **yes**, no share-alike |

These rules are implemented in `src/neurolayer/data/catalog.py`. The source of truth for each dataset is its card in [`catalog/datasets/`](../../catalog/datasets/).

> **Every license below still needs human verification.** The research environment could not open most dataset pages directly. A founder, or later counsel, must read each license page and record `verified_by` / `verified_on` in the card. Until then, the gate blocks `benchmark` and `training` use. This is intentional. WP-1.1 in the [plan](../plan/work-packages.md) covers it.

## 3.1 Motor imagery and execution (CAP-1 candidates)

| Catalog id | Dataset | Subjects | Channels / rate | Sessions | Classes | License (reported) | Likely role |
|------------|---------|----------|-----------------|----------|---------|--------------------|-------------|
| `physionet_mi` | PhysioNet EEG Motor Movement/Imagery (BCI2000) | 109 | 64 ch (10-10), 160 Hz | 1 (14 runs) | L/R fist, both fists, feet (imagery and execution) | **ODC-By 1.0** (Open Data Commons Attribution) | Development and training (if verified) |
| `cho2017` | Cho et al. 2017, GigaDB 100295 | 52 | 64 ch, 512 Hz | 1 | L/R hand | GigaDB data are typically CC0; the article is CC BY 4.0 *(verify)* | Development and training (if verified) |
| `lee2019_mi` | Lee et al. 2019 (OpenBMI), GigaDB 100542 | 54 | 62 ch, 1000 Hz | 2 | L/R hand | GigaDB *(verify)* | Development and training (if verified) |
| `stieger2021` | Stieger et al. 2021, figshare 13123148 | 62 | 64 ch, 1000 Hz | up to 11 (598 total) | L/R, up/down, 2-D cursor (online, continuous) | *(unknown; verify)* | Longitudinal drift and locked-holdout candidate |
| `dreyer2023` | Dreyer et al. 2023, Zenodo 7554429 / 8089820 | 87 | *(verify)* | 1 | L/R hand | *(unknown; verify)* | Locked-holdout candidate (large, includes user-profile data) |
| `bnci2014_001` | BCI Competition IV 2a (BNCI Horizon) | 9 | 22 ch, 250 Hz | 2 | L/R hand, feet, tongue | **CC BY-ND 4.0** as listed by BNCI *(verify)* | Benchmark only (no-derivatives) |
| `bnci2014_004` | BCI Competition IV 2b (BNCI Horizon) | 9 | 3 bipolar (C3, Cz, C4), 250 Hz | 5 | L/R hand | BNCI terms *(verify)* | Benchmark only; also an extreme low-channel stress test |
| `schirrmeister2017` | High-Gamma Dataset | 14 | 128 ch, 500 Hz | 1 | L/R hand, feet, rest (**executed**) | *(verify)* | Motor-execution auxiliary |

**Other motor-imagery datasets available through MOABB** (catalog them in WP-1.4): Weibo2014, Zhou2016, Shin2017A, GrosseWentrup2009, Ofner2017, BNCI2015_001, Beetl2021, Liu2024 (stroke). Each adds montage diversity, which is what cross-device generalization needs.

**Shared label.** Nearly all of these include **left hand vs right hand** imagery. That is why CAP-1 starts with that binary task. It is the only label harmonizable across many datasets, montages and amplifiers today.

## 3.2 Large corpora for self-supervised pretraining

| Catalog id | Dataset | Scale | License (reported) | Notes |
|------------|---------|-------|--------------------|-------|
| `tuh_eeg` | Temple University Hospital EEG Corpus (TUEG) | 26,846 recordings (2002–2017); project >60k EEGs | Access requires a signed request form. Terms reportedly permit research and commercialization *(verify the actual agreement)*. | Clinical and heterogeneous; most foundation models pretrain on it |
| `hbn_eeg` | Healthy Brain Network EEG (OpenNeuro ds005505–ds005515) | 2,600+ participants available; 128 ch | **CC BY-SA 4.0**, except a separate not-for-commercial-use release | Share-alike means legal review is needed before training shippable weights |

## 3.3 Other paradigms (CAP-2 and CAP-3; catalog later)

- **SSVEP:** Wang 2016 benchmark (35 subjects, 40 targets), BETA (70 subjects), Nakanishi; MOABB lists 7 SSVEP datasets.
- **P300/ERP:** MOABB lists 15 datasets (Brain Invaders bi2012–bi2015, BNCI P300 sets, Lee2019 ERP).
- **Error-related potentials:** BNCI Horizon ErrP sets. These are small, which is a data gap for CAP-3.

## 3.4 Excluded

| Dataset | Reason |
|---------|--------|
| Meta generic neuromotor interface (sEMG) | **CC-BY-NC-4.0**, not commercial. Kept in the catalog marked `excluded`, so nobody adds it by accident. |
| SEED / DEAP (emotion) | Academic-only terms. Emotion inference is also out of scope (EU AI Act, see [document 5](05-regulation-and-privacy.md)). |

## 3.5 The data gap is also the proprietary-data opportunity

No public dataset gives us **many users × consumer dry-electrode devices × a shared active-control task**. Public motor-imagery data comes from wet, research-grade, 22–128-channel caps. We simulate consumer layouts by channel subsetting (CAP-1 regime R3). That is a proxy: it does not capture dry-electrode noise, fit variability or motion.

So the proprietary dataset we should build (roadmap Stage 8) is:

- **N ≥ 100** people (hundreds, eventually)
- **2 or more consumer devices** (for example Neurosity Crown and OpenBCI Cyton at sensorimotor sites)
- the **same calibration protocol** we use for CAP-1
- consent language that **explicitly permits commercial model training**
- provenance recorded in the catalog like any other dataset

This mirrors Meta's EMG playbook at startup scale. The recording software should be the product's own calibration flow, so data collection gets cheaper as the product grows.
