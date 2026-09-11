"""Export and verify a portable checkpoint bundle without stopping training.

Copies an open checkpoint snapshot, its tokenizer, and a checksum manifest.
The destination must be new. No source files or remote services are modified.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import torch
from artifact_io import atomic_json, file_sha256
from tokenizer import Tokenizer


FILES = ('latest.pt', 'tokenizer.json')


def checkpoint_summary(directory):
    # The staged file is immutable. mmap avoids reading optimizer tensors into RAM.
    saved = torch.load(Path(directory) / 'latest.pt', map_location='cpu', weights_only=True, mmap=True)
    tok = Tokenizer().load(Path(directory) / 'tokenizer.json')
    if tok.vocab_size != saved['config']['vocab']:
        raise ValueError('Checkpoint and tokenizer vocabularies differ.')
    fingerprint = tok.fingerprint()
    if saved.get('tokenizer_sha256', fingerprint) != fingerprint:
        raise ValueError('Checkpoint and tokenizer merge rules differ.')
    return dict(step=saved.get('step'), stage=saved.get('stage'), config=saved['config'],
                data_sha256=saved.get('data_sha256'), tokenizer_sha256=fingerprint,
                tokenizer_identity_recorded='tokenizer_sha256' in saved,
                optimizer_present='optimizer' in saved, loss_mode=saved.get('loss_mode', 'all'))


def verify_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('version') != 1 or set(manifest.get('files', {})) != set(FILES):
        raise ValueError('Unsupported or incomplete bundle manifest.')
    for name in FILES:
        path = directory / name
        expected = manifest['files'][name]
        if path.is_symlink() or path.stat().st_size != expected['bytes'] or file_sha256(path) != expected['sha256']:
            raise ValueError(f'Bundle integrity check failed: {name}')
    if checkpoint_summary(directory) != manifest['checkpoint']:
        raise ValueError('Checkpoint summary differs from manifest.')
    return manifest


def export_bundle(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('Destination exists; choose a new bundle directory.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + destination.name + '-', dir=destination.parent))
    try:
        tokenizer_bytes = (source / 'tokenizer.json').read_bytes()
        (temporary / 'tokenizer.json').write_bytes(tokenizer_bytes)
        # Opening once pins the old inode if train.py atomically replaces latest.pt.
        with (source / 'latest.pt').open('rb') as src, (temporary / 'latest.pt').open('wb') as dst:
            before = os.fstat(src.fileno())
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
            after = os.fstat(src.fileno())
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Source checkpoint was modified in place during export; retry after saving.')
        if (source / 'tokenizer.json').read_bytes() != tokenizer_bytes:
            raise ValueError('Source tokenizer changed during export; retry with a consistent run.')
        manifest = dict(version=1, created_at=datetime.now(timezone.utc).isoformat(),
                        checkpoint=checkpoint_summary(temporary),
                        files={name: dict(bytes=(temporary / name).stat().st_size,
                                          sha256=file_sha256(temporary / name)) for name in FILES})
        atomic_json(temporary / 'manifest.json', manifest)
        verify_bundle(temporary)
        if destination.exists():
            raise ValueError('Destination was created during export; choose another directory.')
        temporary.rename(destination)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    export.add_argument('--source', type=Path, default=Path('checkpoints'))
    export.add_argument('--destination', type=Path, required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('directory', type=Path)
    args = parser.parse_args()
    try:
        manifest = export_bundle(args.source, args.destination) if args.command == 'export' else verify_bundle(args.directory)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
