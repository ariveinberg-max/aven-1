"""H4: subject-identity decodability of learned features vs tangent space (synthetic).

Fits the spatial-field decoder on the synthetic two-lab pool, then cross-validates a
linear subject probe on (a) its learned features and (b) tangent-space features.
Prints a markdown table for the experiment card. Synthetic data only.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neurolayer.data.sources import load_epochs
from neurolayer.data.synthetic import SyntheticMIConfig
from neurolayer.models.proprietary import SpatialFieldDecoder
from neurolayer.representation.covariance import TangentSpaceEncoder


def _probe(features: np.ndarray, groups: np.ndarray, seed: int = 0) -> tuple[float, float]:
    _, y = np.unique(groups, return_inverse=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    accuracy = float(np.mean(cross_val_score(classifier, features, y, cv=cv)))
    return accuracy, float(np.bincount(y).max() / len(y))


def main() -> int:
    """Run the probe comparison and print a markdown table."""
    config = SyntheticMIConfig(
        n_subjects=8, n_trials_per_class=70, signal_to_noise=0.5, subject_variability=1.0, seed=3
    )
    epochs = load_epochs(["synthetic_mi", "synthetic_mi_shifted"], config)
    groups = np.array([f"{d}/{s}" for d, s in zip(epochs.dataset, epochs.subject, strict=True)])

    decoder = SpatialFieldDecoder(device="cpu")
    decoder.fit(epochs)
    learned = decoder.embed(epochs)
    tangent = TangentSpaceEncoder()
    tangent.fit(epochs)

    print("| representation | dims | subject-probe accuracy | chance | above chance |")
    print("|---|---|---|---|---|")
    for name, features in (
        ("tangent space (no alignment)", tangent.transform(epochs)),
        ("spatial-field features (per-subject aligned)", learned),
        ("spatial-field features (unaligned)", decoder.embed(epochs, align_per_subject=False)),
    ):
        accuracy, chance = _probe(features, groups)
        print(
            f"| {name} | {features.shape[1]} | {accuracy:.3f} | {chance:.3f} | "
            f"{accuracy - chance:+.3f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
