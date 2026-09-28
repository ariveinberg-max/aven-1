"""Turn decoded intents into actions: printing, key presses, agent tools (WP-7.4, 7.5).

Sinks are composable. :class:`DebouncedSink` is what makes a noisy decoder usable: it
only fires when the same intent is decoded confidently several times in a row, then
waits a refractory period. Only decoded **labels** reach a sink, never neural data.

Operating-system input goes through :class:`KeyboardSink` (virtual key presses via
``pynput``). A native Bluetooth HID device, including Apple's BCI HID profile, is a
separate prototype (``docs/research/07-os-input-bci-hid.md``).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Intent:
    """A decoded intent: the winning label and its class probabilities."""

    label: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)


class IntentSink(Protocol):
    """Receives intents as they are decoded."""

    def __call__(self, intent: Intent) -> None:
        """Handle one intent."""
        ...


class PrintSink:
    """Writes each intent to a text callback (stdout by default)."""

    def __init__(self, write: Callable[[str], None] = print) -> None:
        self._write = write

    def __call__(self, intent: Intent) -> None:
        """Print ``label (confidence)``."""
        self._write(f"{intent.label:<12} {intent.confidence:.2f}")


class DebouncedSink:
    """Forward an intent only after ``min_consecutive`` confident repeats.

    Parameters
    ----------
    inner
        Sink that receives the accepted intents.
    threshold
        Minimum confidence for a decode to count.
    min_consecutive
        Required run length of the same confident label.
    refractory_s
        Ignore everything for this long after firing.
    clock
        Time source (injectable for tests).
    """

    def __init__(
        self,
        inner: IntentSink,
        threshold: float = 0.7,
        min_consecutive: int = 2,
        refractory_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if min_consecutive < 1:
            raise ValueError("min_consecutive must be >= 1")
        self._inner = inner
        self._threshold = threshold
        self._needed = min_consecutive
        self._refractory = refractory_s
        self._clock = clock
        self._run_label: str | None = None
        self._run = 0
        self._blocked_until = float("-inf")

    def __call__(self, intent: Intent) -> None:
        """Count the intent and forward it once the run is long enough."""
        now = self._clock()
        if now < self._blocked_until:
            return
        if intent.confidence < self._threshold:
            self._run_label, self._run = None, 0
            return
        if intent.label == self._run_label:
            self._run += 1
        else:
            self._run_label, self._run = intent.label, 1
        if self._run >= self._needed:
            self._inner(intent)
            self._run_label, self._run = None, 0
            self._blocked_until = now + self._refractory


class ActionSink:
    """Map intent labels to callables: app commands or agent tools (WP-7.5).

    Unknown labels are ignored. The callables receive the intent (label and
    probabilities only).
    """

    def __init__(self, actions: Mapping[str, Callable[[Intent], None]]) -> None:
        self._actions = dict(actions)

    def __call__(self, intent: Intent) -> None:
        """Run the action bound to ``intent.label``, if any."""
        action = self._actions.get(intent.label)
        if action is not None:
            action(intent)


DEFAULT_KEYS: dict[str, str] = {"left_hand": "left", "right_hand": "right"}
"""Intent label → key name (``pynput.keyboard.Key`` attribute or a single character)."""


class KeyboardSink:
    """Press a key per intent: controls any app that accepts arrow keys.

    ``press`` is injectable; by default it uses ``pynput`` (install it separately; on
    Linux it needs an X or uinput session, on macOS the Accessibility permission).
    """

    def __init__(
        self,
        keys: Mapping[str, str] | None = None,
        press: Callable[[str], None] | None = None,
    ) -> None:
        self._keys = dict(DEFAULT_KEYS if keys is None else keys)
        self._press = press or _pynput_press()

    def __call__(self, intent: Intent) -> None:
        """Tap the key mapped to ``intent.label``."""
        key = self._keys.get(intent.label)
        if key is not None:
            self._press(key)


def _pynput_press() -> Callable[[str], None]:
    try:
        from pynput.keyboard import Controller, Key
    except ImportError as exc:
        raise ImportError(
            "KeyboardSink needs pynput (`uv pip install pynput`), or pass press=..."
        ) from exc
    controller: Any = Controller()

    def press(name: str) -> None:
        key = getattr(Key, name, None) if len(name) > 1 else name
        if key is None:
            raise ValueError(f"unknown key {name!r}")
        controller.tap(key)

    return press
