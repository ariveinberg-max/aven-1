from __future__ import annotations

import numpy as np
import pytest

from neurolayer.core.types import Recording
from neurolayer.data.audit import audit_dataset, audit_markdown
from neurolayer.data.qa import qa_markdown, qa_recording
from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_recordings

CROWN = ("CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4")


def _recordings(channels: tuple[str, ...] = CROWN, **kwargs: object) -> list[Recording]:
    config = SyntheticMIConfig(n_subjects=2, n_trials_per_class=12, channels=channels, **kwargs)  # type: ignore[arg-type]
    return generate_synthetic_recordings(config)


def _replace_data(recording: Recording, data: np.ndarray) -> Recording:
    return Recording(
        data=data,
        sfreq=recording.sfreq,
        ch_names=recording.ch_names,
        dataset_id=recording.dataset_id,
        subject_id=recording.subject_id,
        session_id=recording.session_id,
        run_id=recording.run_id,
        events=recording.events,
    )


def test_audit_counts_trials_and_montage_coverage() -> None:
    audit = audit_dataset("synthetic_mi", _recordings(n_sessions=2))
    assert len(audit.subjects) == 2
    assert audit.subjects[0].n_sessions == 2
    assert audit.subjects[0].trials_per_label == {"left_hand": 24, "right_hand": 24}
    assert audit.min_trials_per_class() == 24
    assert audit.max_budget(min_test_per_class=10) == 14
    assert "neurosity_crown" in audit.supported_montages()
    assert "muse_s" not in audit.supported_montages()
    report = audit_markdown([audit])
    assert "| synthetic_mi | 2 |" in report
    assert "`muse_s` missing: TP9" in report


def test_audit_rejects_mixed_datasets() -> None:
    with pytest.raises(ValueError, match="in audit of"):
        audit_dataset("other", _recordings())
    with pytest.raises(ValueError, match="no recordings"):
        audit_dataset("x", [])


def test_qa_clean_recording_has_no_issues() -> None:
    qa = qa_recording(_recordings()[0])
    assert qa.issues == []
    assert qa.event_counts == {"left_hand": 12, "right_hand": 12}


def test_qa_flags_flat_noisy_line_noise_and_units() -> None:
    base = _recordings(sfreq=256.0)[0]
    data = np.array(base.data)
    data[0] = 0.0  # flat electrode
    data[1] *= 50.0  # noisy electrode
    t = np.arange(data.shape[1]) / base.sfreq
    data[2:] += 2e-4 * np.sin(2 * np.pi * 50.0 * t)  # strong mains interference
    qa = qa_recording(_replace_data(base, data))
    assert qa.flat_channels == ("CP3",)
    assert qa.noisy_channels == ("C3",)
    assert qa.line_noise_ratio["50"] is not None
    assert qa.line_noise_ratio["50"] > 10
    assert any("line noise at 50 Hz" in issue for issue in qa.issues)

    microvolts_as_volts = _replace_data(base, np.array(base.data) * 1e6)
    assert any("implausible amplitude" in i for i in qa_recording(microvolts_as_volts).issues)
    assert "flagged" in qa_markdown([qa])


def test_line_noise_skipped_above_nyquist() -> None:
    qa = qa_recording(_recordings(sfreq=100.0)[0])
    assert qa.line_noise_ratio == {"50": None, "60": None}
