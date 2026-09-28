"""Dataset adapters: catalogued datasets → canonical recordings (WP-1.2)."""

from neurolayer.data.adapters.base import (
    AdapterNotAvailableError,
    DatasetAdapter,
    IngestionLog,
    get_adapter,
    pseudonymize,
)

__all__ = [
    "AdapterNotAvailableError",
    "DatasetAdapter",
    "IngestionLog",
    "get_adapter",
    "pseudonymize",
]
