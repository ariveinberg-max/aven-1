"""neurolayer: the neural intelligence layer.

Stages (see ``docs/architecture/overview.md``):

1. ``neurolayer.data``: ingestion into canonical :class:`~neurolayer.core.types.Recording`.
2. ``neurolayer.signal``: deterministic, versioned signal processing and epoching.
3. ``neurolayer.representation``: encoders producing neural representations.
4. ``neurolayer.models``: task heads and calibration-time adaptation.

``neurolayer.evaluation`` implements the CAP-1 calibration-efficiency protocol and
``neurolayer.tracking`` records reproducible run manifests.
"""

__version__ = "0.1.0"
