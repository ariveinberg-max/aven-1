from __future__ import annotations

import pytest

from neurolayer.devices.output import ActionSink, DebouncedSink, Intent, KeyboardSink, PrintSink

LEFT = Intent("left_hand", 0.9, {"left_hand": 0.9, "right_hand": 0.1})
RIGHT = Intent("right_hand", 0.8, {"left_hand": 0.2, "right_hand": 0.8})
UNSURE = Intent("right_hand", 0.55, {"left_hand": 0.45, "right_hand": 0.55})


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_debounce_needs_consecutive_confident_repeats() -> None:
    fired: list[Intent] = []
    clock = Clock()
    sink = DebouncedSink(
        fired.append, threshold=0.7, min_consecutive=2, refractory_s=1.0, clock=clock
    )
    for intent in (LEFT, RIGHT, LEFT, UNSURE, LEFT):
        sink(intent)
        clock.now += 0.1
    assert fired == []
    sink(LEFT)
    assert fired == [LEFT]
    sink(RIGHT)
    sink(RIGHT)
    assert fired == [LEFT]  # refractory period
    clock.now += 1.0
    sink(RIGHT)
    sink(RIGHT)
    assert fired == [LEFT, RIGHT]


def test_debounce_validates_parameters() -> None:
    with pytest.raises(ValueError, match="threshold"):
        DebouncedSink(print, threshold=1.5)
    with pytest.raises(ValueError, match="min_consecutive"):
        DebouncedSink(print, min_consecutive=0)


def test_action_and_keyboard_sinks() -> None:
    calls: list[str] = []
    ActionSink({"left_hand": lambda i: calls.append(f"action:{i.label}")})(LEFT)
    ActionSink({"left_hand": lambda i: calls.append("never")})(RIGHT)
    keys = KeyboardSink(press=calls.append)
    keys(LEFT)
    keys(RIGHT)
    KeyboardSink(keys={"left_hand": "a"}, press=calls.append)(RIGHT)
    assert calls == ["action:left_hand", "left", "right"]


def test_print_sink_formats() -> None:
    lines: list[str] = []
    PrintSink(lines.append)(LEFT)
    assert lines == ["left_hand    0.90"]
