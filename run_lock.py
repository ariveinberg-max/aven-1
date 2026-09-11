"""Cross-process advisory checkpoint-writer lock (local filesystems)."""
import os
from pathlib import Path


class WriterBusy(RuntimeError):
    pass


def _lock(file):
    file.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(file):
    file.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


class WriterLock:
    """Keep the lock file: unlinking it creates an inode race between writers.

    OS locks release when a process exits, including crashes. File existence is
    never interpreted as ownership. Every participating writer must use this lock.
    """
    def __init__(self, output):
        self.path = Path(output) / '.writer.lock'
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        try:
            _lock(self.file)
        except OSError as exc:
            self.file.close()
            self.file = None
            raise WriterBusy(f'Another process owns checkpoint directory {self.path.parent}. '
                             'Use a different --output-dir or wait for that writer to finish.') from exc
        # Windows locks byte zero even past EOF. Initialize after acquiring so
        # competing processes never both write it. Later acquisitions are read-only.
        if self.path.stat().st_size == 0:
            self.file.write(b'0')
            self.file.flush()
        return self

    def __exit__(self, *_):
        if self.file is not None:
            try:
                _unlock(self.file)
            finally:
                self.file.close()
                self.file = None


def writer_active(output):
    """Probe without creating directories or changing checkpoint files."""
    path = Path(output) / '.writer.lock'
    try:
        file = path.open('r+b')
    except FileNotFoundError:
        return False
    with file:
        try:
            _lock(file)
        except OSError:
            return True
        _unlock(file)
        return False
