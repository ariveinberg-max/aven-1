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


def allow_safe_rng_globals():
    """Allowlist the numpy/torch globals needed to weights_only=True-load a
    checkpoint's saved RNG state. Different torch builds are stricter than
    others about which globals that requires -- seen needing
    numpy._core.multiarray._reconstruct/numpy.ndarray on one machine's torch,
    and torch._utils._rebuild_device_tensor_from_numpy on another (the
    isolated DirectML venv, an older torch pinned for AMD GPU support), for
    the identical checkpoint format that loads fine elsewhere. PyTorch's own
    documented remediation: allowlist specific, understood functions, not
    disable the safety check. Call before any weights_only=True torch.load
    of one of this project's own checkpoints."""
    import torch
    import _codecs
    import collections
    globals_to_allow = [_codecs.encode, collections.OrderedDict]
    try:
        import numpy
        globals_to_allow += [
            numpy._core.multiarray._reconstruct, numpy.ndarray, numpy.dtype,
            numpy.dtypes.Float64DType, numpy.dtypes.Float32DType, numpy.dtypes.Int64DType,
            numpy.dtypes.UInt8DType, numpy.dtypes.Int32DType, numpy.dtypes.BoolDType,
        ]
        # Older numpy exposes the same reconstruct/scalar helpers under a
        # different module path -- harmless to also allow if present.
        legacy = getattr(numpy, 'core', None)
        if legacy is not None and hasattr(legacy, 'multiarray'):
            globals_to_allow.append(legacy.multiarray._reconstruct)
    except Exception:
        pass
    try:
        globals_to_allow.append(torch._utils._rebuild_device_tensor_from_numpy)
    except AttributeError:
        pass
    if globals_to_allow:
        try:
            torch.serialization.add_safe_globals(globals_to_allow)
        except AttributeError:
            pass


def load_checkpoint_snapshot(path, expected_sha256=None):
    """Hash and load the same open file despite atomic path replacement."""
    import torch
    allow_safe_rng_globals()
    with Path(path).open('rb') as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
        fingerprint = digest.hexdigest()
        if expected_sha256 is not None and fingerprint != expected_sha256:
            raise ValueError('Checkpoint SHA-256 mismatch.')
        source.seek(0)
        return torch.load(source, map_location='cpu', weights_only=True), fingerprint
