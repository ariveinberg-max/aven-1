"""SQLite-backed notes storage. Not part of the neural network.

This is a plain database table you read, write, and delete directly. Saving
a note here never changes the model's weights, and training never reads
this database. Explicit /recall requests retrieve excerpts through recall.py. It exists so you have a place
to keep facts, snippets, or reminders that is separate from what the
network has learned, and separate from the raw training-data workspace.
"""
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / 'memory.db'


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL DEFAULT '',
        content TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )''')
    return conn


def list_records():
    with connect() as conn:
        rows = conn.execute('SELECT id, title, content, created_at, updated_at FROM memory ORDER BY updated_at DESC').fetchall()
    return [dict(id=r[0], title=r[1], content=r[2], created_at=r[3], updated_at=r[4]) for r in rows]


def add_record(title, content):
    title, content = title.strip(), content.strip()
    if not content:
        raise ValueError('A memory record needs content.')
    if len(content) > 20000:
        raise ValueError('Keep a single record under 20,000 characters.')
    with connect() as conn:
        cur = conn.execute('INSERT INTO memory (title, content) VALUES (?, ?)', (title[:200], content))
        return cur.lastrowid


def update_record(record_id, title, content):
    title, content = title.strip(), content.strip()
    if not content:
        raise ValueError('A memory record needs content.')
    if len(content) > 20000:
        raise ValueError('Keep a single record under 20,000 characters.')
    with connect() as conn:
        cur = conn.execute(
            "UPDATE memory SET title=?, content=?, updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=?",
            (title[:200], content, record_id))
        if cur.rowcount == 0:
            raise ValueError('That record no longer exists.')


def delete_record(record_id):
    with connect() as conn:
        cur = conn.execute('DELETE FROM memory WHERE id=?', (record_id,))
        if cur.rowcount == 0:
            raise ValueError('That record no longer exists.')
