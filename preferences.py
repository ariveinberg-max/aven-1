"""SQLite storage for human preference comparisons — the raw material for RLHF.

Each row is one real decision: given a prompt and two model-generated
responses, which one did a person actually prefer. This is not synthetic —
faking these labels would defeat the entire point of RLHF (learning what a
human actually wants, not what the model already thinks is good).
"""
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / 'preferences.db'


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS preferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt TEXT NOT NULL,
        response_a TEXT NOT NULL,
        response_b TEXT NOT NULL,
        winner TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'human',
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )''')
    # Migration for databases created before the source column existed --
    # ALTER TABLE ADD COLUMN with a default backfills existing rows too.
    existing = {row[1] for row in conn.execute('PRAGMA table_info(preferences)')}
    if 'source' not in existing:
        conn.execute("ALTER TABLE preferences ADD COLUMN source TEXT NOT NULL DEFAULT 'human'")
    return conn


def add(prompt, response_a, response_b, winner, source='human'):
    if winner not in ('a', 'b', 'tie'):
        raise ValueError('winner must be "a", "b", or "tie".')
    if source not in ('human', 'ai'):
        raise ValueError('source must be "human" or "ai" -- never blend without recording which.')
    with connect() as conn:
        cur = conn.execute(
            'INSERT INTO preferences (prompt, response_a, response_b, winner, source) VALUES (?, ?, ?, ?, ?)',
            (prompt, response_a, response_b, winner, source))
        return cur.lastrowid


def count():
    with connect() as conn:
        return conn.execute('SELECT COUNT(*) FROM preferences').fetchone()[0]


def list_recent(limit=30):
    with connect() as conn:
        rows = conn.execute(
            'SELECT id, prompt, response_a, response_b, winner, source, created_at FROM preferences ORDER BY id DESC LIMIT ?',
            (limit,)).fetchall()
    return [dict(id=r[0], prompt=r[1], response_a=r[2], response_b=r[3], winner=r[4], source=r[5], created_at=r[6]) for r in rows]


def delete(record_id):
    with connect() as conn:
        cur = conn.execute('DELETE FROM preferences WHERE id=?', (record_id,))
        if cur.rowcount == 0:
            raise ValueError('That record no longer exists.')


def all_decided():
    """Non-tie comparisons, for reward model training."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT prompt, response_a, response_b, winner, source FROM preferences WHERE winner != 'tie'").fetchall()
    return [dict(prompt=r[0], response_a=r[1], response_b=r[2], winner=r[3], source=r[4]) for r in rows]


def count_by_source():
    """Real provenance breakdown -- never report a total without this alongside it."""
    with connect() as conn:
        rows = conn.execute('SELECT source, COUNT(*) FROM preferences GROUP BY source').fetchall()
    return dict(rows)
