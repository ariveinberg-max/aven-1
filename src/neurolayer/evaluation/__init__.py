"""CAP-1 evaluation harness: folds, calibration splits, metrics and the protocol runner.

**Protected module.** Specification: ``docs/architecture/evaluation-protocol.md``.
Changes to splitting, metrics or the protocol loop require an ADR (ADR-0004).
"""

from neurolayer.evaluation.protocol import (
    BudgetSummary,
    ProtocolConfig,
    ProtocolResult,
    SkippedSubject,
    SubjectResult,
    run_protocol,
)
from neurolayer.evaluation.splits import (
    CalibrationPlan,
    Fold,
    InsufficientTrialsError,
    LeakageError,
    leave_dataset_out_folds,
    plan_calibration,
    within_dataset_folds,
)

__all__ = [
    "BudgetSummary",
    "CalibrationPlan",
    "Fold",
    "InsufficientTrialsError",
    "LeakageError",
    "ProtocolConfig",
    "ProtocolResult",
    "SkippedSubject",
    "SubjectResult",
    "leave_dataset_out_folds",
    "plan_calibration",
    "run_protocol",
    "within_dataset_folds",
]
