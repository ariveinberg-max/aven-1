"""Training-data workspace: named text snippets kept separate from the
compiled corpus. Combine them into data/training.txt when you are ready to
start a new training run. This never touches model weights or the memory
database; it only manages the plain text files a new run would learn from.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES_DIR = ROOT / 'data/sources'
CORPUS_PATH = ROOT / 'data/training.txt'
NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}$')


def _safe_path(name):
    if not NAME_RE.match(name or ''):
        raise ValueError('Names may use letters, numbers, spaces, dots, hyphens, underscores (max 80 characters).')
    path = (SOURCES_DIR / name).resolve()
    if path.parent != SOURCES_DIR.resolve():
        raise ValueError('Invalid name.')
    return path


def list_sources():
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(SOURCES_DIR.glob('*')):
        if path.is_file():
            stat = path.stat()
            items.append(dict(name=path.name, bytes=stat.st_size, updated_at=stat.st_mtime))
    return items


def read_source(name):
    path = _safe_path(name)
    if not path.exists():
        raise ValueError('That source no longer exists.')
    return path.read_text(encoding='utf-8', errors='replace')


def add_or_update_source(name, text):
    text = text.strip()
    if not text:
        raise ValueError('A source needs some text.')
    if len(text.encode('utf-8')) > 5_000_000:
        raise ValueError('Keep a single source under 5 MB; split large corpora into several sources.')
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    path = _safe_path(name)
    path.write_text(text, encoding='utf-8')


def delete_source(name):
    path = _safe_path(name)
    if not path.exists():
        raise ValueError('That source no longer exists.')
    path.unlink()


def compile_corpus():
    """Stream sources into data/training.txt in chunks rather than reading
    every source fully into memory and joining as one big Python string --
    at hundreds of MB to GB (e.g. a Wikipedia extraction source), the old
    approach could itself need as much RAM as the corpus is large, on top
    of everything else."""
    items = list_sources()
    if not items:
        raise ValueError('Add at least one source to the workspace before compiling.')
    total_size = sum(item['bytes'] for item in items)
    if total_size > 2_000_000_000:
        raise ValueError('Combined sources exceed the 2 GB limit.')
    ROOT.joinpath('data').mkdir(exist_ok=True)
    tmp = CORPUS_PATH.with_suffix('.tmp')
    written = 0
    with open(tmp, 'wb') as out:
        for i, item in enumerate(items):
            with open(SOURCES_DIR / item['name'], 'rb') as src:
                for chunk in iter(lambda: src.read(8 * 1024 * 1024), b''):
                    out.write(chunk)
                    written += len(chunk)
            if i < len(items) - 1:
                out.write(b'\n\n')
                written += 2
    if written < 4096:
        tmp.unlink()
        raise ValueError('Combined sources must total at least 4 KB (roughly 700-1000 words).')
    tmp.replace(CORPUS_PATH)
    return written
