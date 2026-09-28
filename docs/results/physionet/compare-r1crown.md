# Run comparison

| run | decoder | BA@0 | BA@5 | BA@8 | UUR@max | AUCEC | median TTC |
|---|---|---|---|---|---|---|---|
| `20260928T031340Z-physionet-b0-r1crown-0441f4` | chance · neurosity_crown | 0.495 [0.48, 0.51] | 0.495 [0.48, 0.51] | 0.495 [0.48, 0.51] | 0.00 | 0.495 | never |
| `20260928T031258Z-physionet-b1-r1crown-516f62` | csp_lda_subject · neurosity_crown | 0.500 [0.50, 0.50] | 0.550 [0.53, 0.57] | 0.582 [0.56, 0.60] | 0.18 | 0.533 | never |
| `20260928T031322Z-physionet-b2-r1crown-cfd78e` | ts_lr_subject · neurosity_crown | 0.500 [0.50, 0.50] | 0.569 [0.55, 0.59] | 0.582 [0.56, 0.61] | 0.19 | 0.542 | never |
| `20260928T015255Z-physionet-b3-r1crown-5bc0b8` | ts_lr_pooled · neurosity_crown | 0.626 [0.60, 0.65] | 0.612 [0.59, 0.64] | 0.629 [0.60, 0.65] | 0.27 | 0.619 | never |
| `20260928T031833Z-physionet-b4-r1crown-179f8a` | braindecode (EEGNet) · neurosity_crown | 0.756 [0.73, 0.78] | 0.760 [0.74, 0.78] | 0.763 [0.74, 0.79] | 0.64 | 0.759 | 0 |
| `20260928T031229Z-physionet-nl-r1crown-a8d4e7` | nl_spatial_field · neurosity_crown | 0.645 [0.62, 0.67] | 0.618 [0.59, 0.64] | 0.630 [0.60, 0.66] | 0.31 | 0.630 | never |

## Paired vs reference: ts_lr_pooled · neurosity_crown (`20260928T015255Z-physionet-b3-r1crown-5bc0b8`)

| decoder | k | mean ΔBA | Holm p | n subjects |
|---|---|---|---|---|
| chance · neurosity_crown | 0 | -0.130 | 6.53e-14 | 109 |
| chance · neurosity_crown | 5 | -0.117 | 2.07e-12 | 109 |
| chance · neurosity_crown | 8 | -0.133 | 6.53e-14 | 109 |
| csp_lda_subject · neurosity_crown | 0 | -0.126 | 9.57e-15 | 109 |
| csp_lda_subject · neurosity_crown | 5 | -0.062 | 6.43e-07 | 109 |
| csp_lda_subject · neurosity_crown | 8 | -0.047 | 2.4e-05 | 109 |
| ts_lr_subject · neurosity_crown | 0 | -0.126 | 9.57e-15 | 109 |
| ts_lr_subject · neurosity_crown | 5 | -0.044 | 1.63e-05 | 109 |
| ts_lr_subject · neurosity_crown | 8 | -0.046 | 1.31e-05 | 109 |
| braindecode (EEGNet) · neurosity_crown | 0 | +0.130 | 7.36e-15 | 109 |
| braindecode (EEGNet) · neurosity_crown | 5 | +0.148 | 1.27e-14 | 109 |
| braindecode (EEGNet) · neurosity_crown | 8 | +0.134 | 1.45e-13 | 109 |
| nl_spatial_field · neurosity_crown | 0 | +0.019 | 0.109 | 109 |
| nl_spatial_field · neurosity_crown | 5 | +0.005 | 1 | 109 |
| nl_spatial_field · neurosity_crown | 8 | +0.001 | 1 | 109 |

![Calibration-efficiency curves](cec-r1crown.svg)
