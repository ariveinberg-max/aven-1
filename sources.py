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
    items = list_sources()
    if not items:
        raise ValueError('Add at least one source to the workspace before compiling.')
    parts = [Path(SOURCES_DIR / item['name']).read_text(encoding='utf-8', errors='replace') for item in items]
    combined = '\n\n'.join(parts).strip() + '\n'
    raw = combined.encode('utf-8')
    if len(raw) < 4096:
        raise ValueError('Combined sources must total at least 4 KB (roughly 700-1000 words).')
    if len(raw) > 20_000_000:
        raise ValueError('Combined sources exceed the 20 MB starter limit.')
    ROOT.joinpath('data').mkdir(exist_ok=True)
    CORPUS_PATH.write_bytes(raw)
    return len(raw)
