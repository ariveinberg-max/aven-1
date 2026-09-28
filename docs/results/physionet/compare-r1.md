# Run comparison

| run | decoder | BA@0 | BA@5 | BA@8 | UUR@max | AUCEC | median TTC |
|---|---|---|---|---|---|---|---|
| `20260928T012419Z-physionet-b0-r1-b23620` | chance | 0.495 [0.48, 0.51] | 0.495 [0.48, 0.51] | 0.495 [0.48, 0.51] | 0.00 | 0.495 | never |
| `20260928T045018Z-physionet-b1-r1-b90798` | csp_lda_subject | 0.500 [0.50, 0.50] | 0.546 [0.53, 0.57] | 0.553 [0.53, 0.57] | 0.10 | 0.528 | never |
| `20260928T045122Z-physionet-b2-r1-bee681` | ts_lr_subject | 0.500 [0.50, 0.50] | 0.561 [0.54, 0.58] | 0.572 [0.55, 0.59] | 0.15 | 0.537 | never |
| `20260928T050420Z-physionet-b3-r1-41b7a1` | ts_lr_pooled | 0.607 [0.59, 0.63] | 0.613 [0.59, 0.64] | 0.619 [0.60, 0.65] | 0.26 | 0.611 | never |
| `20260928T050550Z-physionet-b4-r1-0b37e4` | braindecode (EEGNet) | 0.760 [0.73, 0.79] | 0.764 [0.74, 0.79] | 0.772 [0.75, 0.80] | 0.67 | 0.763 | 0 |
| `20260928T040417Z-physionet-nl-r1-b80175` | nl_spatial_field | 0.698 [0.67, 0.73] | 0.666 [0.64, 0.69] | 0.695 [0.67, 0.72] | 0.48 | 0.682 | 0 |

## Paired vs reference: ts_lr_pooled (`20260928T050420Z-physionet-b3-r1-41b7a1`)

| decoder | k | mean ΔBA | Holm p | n subjects |
|---|---|---|---|---|
| chance | 0 | -0.112 | 4.11e-13 | 109 |
| chance | 5 | -0.118 | 4.11e-13 | 109 |
| chance | 8 | -0.124 | 4.11e-13 | 109 |
| csp_lda_subject | 0 | -0.107 | 1.33e-12 | 109 |
| csp_lda_subject | 5 | -0.067 | 1.11e-05 | 109 |
| csp_lda_subject | 8 | -0.066 | 3.85e-06 | 109 |
| ts_lr_subject | 0 | -0.107 | 1.33e-12 | 109 |
| ts_lr_subject | 5 | -0.052 | 0.000156 | 109 |
| ts_lr_subject | 8 | -0.048 | 0.000431 | 109 |
| braindecode (EEGNet) | 0 | +0.152 | 4.12e-16 | 109 |
| braindecode (EEGNet) | 5 | +0.151 | 1.11e-15 | 109 |
| braindecode (EEGNet) | 8 | +0.153 | 4.12e-16 | 109 |
| nl_spatial_field | 0 | +0.091 | 2.29e-10 | 109 |
| nl_spatial_field | 5 | +0.053 | 3.32e-05 | 109 |
| nl_spatial_field | 8 | +0.076 | 3.83e-08 | 109 |

![Calibration-efficiency curves](cec-r1.svg)
