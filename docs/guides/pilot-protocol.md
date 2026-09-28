# Pilot data collection protocol (WP-8.3, Gate 3)

**Status:** draft v0.1, for review by the founder, counsel and (recommended) an independent ethics reviewer before the first participant. Nothing here is legal advice.

## 1. Purpose

Record cued left/right motor imagery from ≥ 20 people on the consumer device chosen in the [device selection memo](../templates/device-selection-memo.md). The data serves two purposes:

1. Evaluate CAP-1 on a real consumer device (**Gate 3**: regime R3 on our own data).
2. With separate consent only, build the first proprietary dataset.

## 2. Participants

- Adults (18+) able to give informed consent themselves.
- Exclude people with a history of epilepsy or photosensitive seizures (the calibration game uses visual cues), and people with skin conditions at electrode sites.
- Target ≥ 20 participants, balanced across sex and age bands where possible. Record handedness (it affects motor-imagery lateralization).
- Compensation: `TODO(founder)`. It must not depend on performance.

## 3. Consent (WP-8.2)

- Use the [consent form template](../templates/consent-form.md) only after counsel review. Record every signature with its version:

  ```bash
  uv run neurolayer consent grant P001 --purposes product,research \
      --document consent-v1.0.pdf --document-version v1.0 --recorded-by <staff>
  ```

- Purposes are separate checkboxes. `commercial_training` is optional, and declining it changes nothing about participation.
- Withdrawal at any time: `neurolayer consent revoke P001 --purposes research ...`, then delete the participant's recordings and log the deletion (PRIV-4).
- The pseudonym (`P001`) is the only identifier in any data file. The key linking names to pseudonyms lives in a separate, access-restricted system.

## 4. Session structure (about 45 minutes)

| Step | Minutes | Notes |
|------|---------|-------|
| Welcome, consent, questionnaire | 10 | Handedness, prior BCI experience, sleep and caffeine today |
| Device fitting and signal check | 5–10 | Use the dashboard's Inspect page or `neurolayer data qa` on a 1-minute test recording; fix flat or noisy channels before continuing |
| Practice (not recorded) | 3 | 5 cues per hand, explain kinesthetic imagery ("feel the squeeze", not "see the hand") |
| Run 1 | 4 | `neurolayer pilot record P001 --session 1 --board <BOARD> --trials-per-class 20` |
| Break | 2 | |
| Run 2 | 4 | `--session 2`, different `--seed` |
| Live feedback (optional, not stored) | 5 | `neurolayer bridge` with the print sink; record only the participant's subjective rating |
| Debrief | 3 | |

Timing per trial: 2 s rest, cue, 4 s imagery (`CueSchedule` defaults). The CAP-1 window (0.5–2.5 s after the cue) fits inside the imagery period.

## 5. Timing validation (once per device, before participant 1)

Cues are placed on the sample clock. They are displayed slightly after the samples they are aligned to, by the device's transport latency. Measure that latency with a photodiode on the cue area, recorded as an extra channel or through a hardware trigger. Accept the device if median latency is < 50 ms and jitter (IQR) is < 20 ms. Otherwise correct event onsets by the measured median and record the correction in the session metadata. `TODO(verify)`: the latency of the chosen device.

## 6. Data handling

- Recordings are written as BIDS-EEG under `<data-root>/pilot/bids`, with a session sidecar (schedule, device, consent-form hash) and checksums.
- `<data-root>` must be on encrypted storage (class C3). Never in git (the data guard blocks it), never in an unencrypted DVC remote, never sent to external APIs (including LLMs).
- Access is logged and limited to named staff.
- Retention: `TODO(counsel)`: raw data retention period, and whether derived models must be retrained after a withdrawal (see ADR-0012).

## 7. Analysis plan (fixed before data collection)

1. Per-recording QA (`neurolayer data qa`). Exclusion rules: > 25% flat or noisy channels, or < 30 usable trials per class. Decide and record exclusions before any decoder is run.
2. Gate 3: run the CAP-1 harness in regime R3 with the models frozen at Gate 2. Report BA@k, UUR@70% and TTC70 with bootstrap CIs. **No retuning on pilot data** before the Gate 3 numbers are recorded.
3. Only participants with `commercial_training` consent may enter any training set, and only after ADR-0012 is accepted.

## 8. Adverse events

Stop the session on any discomfort, headache or distress. Record it in the incident log (no neural data in the log). Serious events go to the founder the same day.
