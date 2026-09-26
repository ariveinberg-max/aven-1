"""Stage 2: deterministic, versioned signal processing and epoching.

Planned contract (WP-2.1): every transform is a pure function ``Recording -> Recording``
with a declared version. A pipeline's hash covers its config and every transform
version and addresses its cached output in ``data/processed/<pipeline_hash>/``.
"""
