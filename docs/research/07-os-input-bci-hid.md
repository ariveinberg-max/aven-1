# 7. OS input: from decoded intent to standard input events (WP-7.4)

**Date:** 2026-09-26 · **Status:** v1 research note. Items marked `TODO(verify)` are unconfirmed and must be checked against primary sources before any build or partner conversation.

## 7.1 Why this matters

A decoder is only useful if its output drives software people already use. There are two routes:

1. **An SDK** that apps integrate. Slow: every app has to adopt it.
2. **Standard input events** (keys, switches, pointer) that the operating system already understands. Every app works on day one.

Stage 7 builds route 2 first. Our decoder's output is a stream of `Intent(label, confidence)`. The bridge turns confident, debounced intents into input events.

## 7.2 What exists

| Route | Platforms | What it gives us | Status in this repo |
|-------|-----------|------------------|---------------------|
| Virtual key presses (`pynput`) | Windows, macOS (needs the Accessibility permission), Linux (X11/uinput) | Arrow keys / any key per intent. Drives games, slide decks, switch-scanning software | **Prototype:** `neurolayer.devices.output.KeyboardSink`, `neurolayer bridge --sink keyboard` |
| OS switch access (Switch Control on Apple platforms, Switch Access on Android) | iOS, iPadOS, macOS, visionOS, Android | Scanning UIs built for one or two switches, which fits a 2-class decoder well | Reachable today by mapping intents to the keys the switch software listens for (e.g. space / enter). `TODO(verify)`: per-platform key bindings |
| **Apple BCI HID** (announced 13 May 2025, [coverage](https://www.technewsworld.com/story/apple-adds-brain-to-computer-protocol-to-its-accessibility-repertoire-179739.html), [Synchron release](https://www.businesswire.com/news/home/20250513927084/en/Synchron-To-Achieve-First-Native-Brain-Computer-Interface-Integration-with-iPhone-iPad-and-Apple-Vision-Pro)) | iOS, iPadOS, visionOS | A Human Interface Device profile for brain-computer interfaces, surfaced through Switch Control. Synchron was first; Cognixion is integrating | **Not built.** See 7.3 |
| Bluetooth LE HID keyboard/switch emulation from our own hardware or a dongle | All platforms | A device that pairs like a keyboard or switch, with no host software | Follow-up; needs hardware |

## 7.3 Apple BCI HID: what we know and what we must verify

Known from the sources above: it is a HID profile; it is exposed through Switch Control; it ships on iOS, iPadOS and visionOS; implant companies integrated first.

Unknown, `TODO(verify)` before committing engineering time:

- How third parties get the specification (public HID usage tables, MFi program, or partnership only).
- Whether non-invasive (EEG) devices are eligible, or only medical implants.
- Whether a software-only bridge (a Mac or phone app relaying decoded intents) can present the profile, or it must be the peripheral itself over Bluetooth.
- Minimum OS versions, and whether macOS is supported.
- What the profile carries: discrete switch events only, or also richer signals (confidence, continuous control) that Switch Control can use.

## 7.4 Design consequences (already reflected in the code)

- **Debouncing is mandatory.** A 70% decoder emits a wrong intent every few seconds. `DebouncedSink` requires `min_consecutive` confident decodes, then a refractory period. Tune both per user in the calibration game before connecting to the OS.
- **Two intents map to "next" and "select"** in switch-scanning UIs. This is the natural first OS target for CAP-1.
- **Asynchronous control is not yet measured.** CAP-1 is cue-locked. The bridge's sliding-window decoding is a product prototype. Measuring self-paced control (false activations per minute at rest) is CAP-1 v2 work and must be done before any accessibility claim.
- **Safety:** input bridges must have an obvious off switch. On Apple platforms the user keeps Switch Control's own controls. For `KeyboardSink`, the bridge runs for a fixed `--duration`.

## 7.5 Next steps

1. Answer the `TODO(verify)` list (Apple developer documentation, then partner contact).
2. Dogfood `neurolayer bridge --sink keyboard` with the simulated headset and then a real device. Log false activations per minute at rest.
3. If a software bridge cannot present BCI HID, evaluate a BLE HID dongle as the Gate 3 companion device.
