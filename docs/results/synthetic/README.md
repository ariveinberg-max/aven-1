# Synthetic baseline suite: pipeline demonstration (NOT results)

These reports come from the **synthetic** generator (`neurolayer.data.synthetic`), two synthetic "labs" with 8 subjects each. They show that the whole Stage 4 toolchain works end to end:

- baseline configs
- the CAP-1 protocol across regimes R1 (new subject), R2 (new dataset) and R3 (restricted montage)
- comparison reports with paired statistics
- calibration-efficiency curves

**They say nothing about real EEG** and must never be cited as capability evidence. Only official runs on real data go into the [results ledger](../README.md).

| Regime | Report | Curves |
|--------|--------|--------|
| R1: new subjects, same datasets | [baselines-r1.md](baselines-r1.md) | [cec-r1.svg](cec-r1.svg) |
| R2: leave-one-dataset-out | [baselines-r2.md](baselines-r2.md) | [cec-r2.svg](cec-r2.svg) |
| R3: R2 + 3-channel target montage | [baselines-r3.md](baselines-r3.md) | [cec-r3.svg](cec-r3.svg) |

Reproduce:

```bash
for r in r1 r2 r3; do for b in b0 b1 b2 b3 b4; do
  uv run neurolayer --output artifacts/runs/synthetic-suite run configs/experiments/baselines/synthetic_${b}_${r}.yaml
done; done
uv run neurolayer report compare artifacts/runs/synthetic-suite/*-b?-r2-* \
  --reference artifacts/runs/synthetic-suite/*-b3-r2-* --plot docs/results/synthetic/cec-r2.svg
```

## What the toolchain shows (qualitatively, on synthetic data)

- **Calibration-only baselines (B1, B2)** score exactly chance at k = 0 by construction, then climb with calibration.
- The **pooled Riemannian baseline with per-subject re-centering (B3)** is strongest at zero calibration. This is the pattern the MOABB benchmark and the transfer-learning literature report for real EEG.
- **EEGNet (B4)** is competitive only with enough source data. Euclidean alignment matters (+6–8 pp in a side check).
- **Label-shuffle control:** the across-seed mean is at chance (0.491–0.509) at every k over 5 seeds (ADR-0010). A single seed can look above chance: that is shared classifier randomness, not leakage.
- **Identity probe:** subject identity is decodable at ≥ 0.96 from every classical representation. Real EEG behaves the same way (the 2026 "identity trap" audits), which is why the probe is reported next to every model result.

## Artifacts not in git

The run directories (`artifacts/runs/...`) are git-ignored. The run ids in the reports identify them on the machine that produced them.
