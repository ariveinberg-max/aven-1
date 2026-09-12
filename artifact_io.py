"""Atomic local artifact writes with unique temporary names and flushed data."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, writer, overwrite=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as output:
            writer(output)
            output.flush()
            os.fsync(output.fileno())
        if overwrite:
            temporary.replace(path)
        else:
            # Publish a complete file only if no destination exists. Hard-linking
            # within the same directory avoids a check-then-overwrite race.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value, overwrite=True):
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
    atomic_write(path, lambda output: output.write(encoded), overwrite=overwrite)


def atomic_torch_save(path, value):
    import torch
    atomic_write(path, lambda output: torch.save(value, output))


def validate_tokenizer(checkpoint, tokenizer, label='Checkpoint'):
    if checkpoint['config']['vocab'] != tokenizer.vocab_size:
        raise ValueError(f'{label} vocabulary does not match the tokenizer.')
    recorded = checkpoint.get('tokenizer_sha256')
    if recorded is not None and recorded != tokenizer.fingerprint():
        raise ValueError(f'{label} merge rules do not match the tokenizer.')
    return recorded is not None


def load_checkpoint_snapshot(path, expected_sha256=None):
    """Hash and load the same open file despite atomic path replacement."""
    import torch
    # Some checkpoints store CPU RNG state as a numpy array (see train.py's
    # random-state save/resume). Newer torch versions' weights_only=True
    # loader rejects numpy's own array-reconstruction function unless it's
    # explicitly allowlisted -- seen failing on the Windows PC's torch build
    # even though the identical checkpoint format loads fine elsewhere.
    # This is PyTorch's own documented remediation for exactly this case:
    # allowlisting one specific, understood function, not disabling the
    # safety check itself.
    try:
        import numpy
        torch.serialization.add_safe_globals([
            numpy._core.multiarray._reconstruct, numpy.ndarray, numpy.dtype,
            numpy.dtypes.Float64DType, numpy.dtypes.Float32DType, numpy.dtypes.Int64DType,
        ])
    except Exception:
        pass  # older torch/numpy without this API, or numpy unavailable -- harmless to skip
    with Path(path).open('rb') as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
        fingerprint = digest.hexdigest()
        if expected_sha256 is not None and fingerprint != expected_sha256:
            raise ValueError('Checkpoint SHA-256 mismatch.')
        source.seek(0)
        return torch.load(source, map_location='cpu', weights_only=True), fingerprint
