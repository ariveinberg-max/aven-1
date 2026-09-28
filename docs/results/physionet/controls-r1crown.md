# Label-shuffle control and identity probe (real data, PhysioNet MI, R1-Crown)

Clean checkout at commit 70c0582. Command: `neurolayer control configs/experiments/baselines/physionet_b3_r1crown.yaml --seeds 5` and `neurolayer probe ... --encoders tangent_space log_variance`.

## Label-shuffle control (ADR-0010, B3, 5 seeds)

| k | mean BA across seeds | SE | compatible with chance |
|---|---|---|---|
| 0 | 0.496 | 0.005 | yes |
| 5 | 0.489 | 0.007 | yes |
| 8 | 0.503 | 0.004 | yes |

## Identity probe (features of the Crown montage)

| encoder | target | classes | probe accuracy | chance | above chance |
|---|---|---|---|---|---|
| tangent_space | subject | 109 | 1.000 | 0.012 | +0.988 |
| log_variance | subject | 109 | 0.941 | 0.012 | +0.930 |
