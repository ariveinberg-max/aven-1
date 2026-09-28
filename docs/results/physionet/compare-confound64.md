# Run comparison

| run | decoder | BA@0 | BA@5 | BA@8 | UUR@max | AUCEC | median TTC |
|---|---|---|---|---|---|---|---|
| `20260928T050700Z-physionet-confound64-b3-motor-67ec73` | ts_lr_pooled | 0.632 [0.61, 0.66] | 0.620 [0.60, 0.65] | 0.626 [0.60, 0.65] | 0.29 | 0.626 | never |
| `20260928T051141Z-physionet-confound64-b3-nonmotor-b9d130` | ts_lr_pooled | 0.589 [0.57, 0.61] | 0.591 [0.57, 0.61] | 0.594 [0.57, 0.62] | 0.20 | 0.590 | never |
| `20260928T052326Z-physionet-confound64-nl-motor-bdc55d` | nl_spatial_field | 0.664 [0.64, 0.69] | 0.645 [0.62, 0.67] | 0.662 [0.64, 0.69] | 0.39 | 0.654 | never |
| `20260928T053536Z-physionet-confound64-nl-nonmotor-be0ea1` | nl_spatial_field | 0.634 [0.61, 0.66] | 0.618 [0.60, 0.64] | 0.636 [0.61, 0.66] | 0.32 | 0.626 | never |
| `20260928T050420Z-physionet-b3-r1-41b7a1` | ts_lr_pooled | 0.607 [0.59, 0.63] | 0.613 [0.59, 0.64] | 0.619 [0.60, 0.65] | 0.26 | 0.611 | never |
| `20260928T040417Z-physionet-nl-r1-b80175` | nl_spatial_field | 0.698 [0.67, 0.73] | 0.666 [0.64, 0.69] | 0.695 [0.67, 0.72] | 0.48 | 0.682 | 0 |

![Calibration-efficiency curves](cec-confound64.svg)
