from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import preferences


class PreferencesTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._patch = patch.object(preferences, 'DB_PATH', Path(self._tmpdir.name) / 'preferences.db')
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmpdir.cleanup()

    def test_add_defaults_to_human_source(self):
        rid = preferences.add('prompt', 'a', 'b', 'a')
        row = preferences.list_recent(1)[0]
        self.assertEqual(row['id'], rid)
        self.assertEqual(row['source'], 'human')

    def test_add_rejects_invalid_winner(self):
        with self.assertRaises(ValueError):
            preferences.add('prompt', 'a', 'b', 'c')

    def test_add_rejects_invalid_source(self):
        with self.assertRaises(ValueError):
            preferences.add('prompt', 'a', 'b', 'a', source='synthetic')

    def test_ai_source_is_tracked_separately_from_human(self):
        preferences.add('p1', 'a', 'b', 'a', source='human')
        preferences.add('p2', 'a', 'b', 'b', source='ai')
        preferences.add('p3', 'a', 'b', 'a', source='ai')
        self.assertEqual(preferences.count_by_source(), {'human': 1, 'ai': 2})

    def test_all_decided_excludes_ties_and_includes_source(self):
        preferences.add('p1', 'a', 'b', 'a', source='human')
        preferences.add('p2', 'a', 'b', 'tie', source='human')
        preferences.add('p3', 'a', 'b', 'b', source='ai')
        decided = preferences.all_decided()
        self.assertEqual(len(decided), 2)
        self.assertEqual({d['source'] for d in decided}, {'human', 'ai'})

    def test_delete_removes_record(self):
        rid = preferences.add('p', 'a', 'b', 'a')
        preferences.delete(rid)
        self.assertEqual(preferences.count(), 0)
        with self.assertRaises(ValueError):
            preferences.delete(rid)

    def test_migration_backfills_existing_rows_without_source_column(self):
        # Simulate a pre-migration database: create the table by hand, without
        # the source column, insert a row the old way, then confirm connect()
        # migrates it in place rather than losing or mislabeling the row.
        import sqlite3
        conn = sqlite3.connect(preferences.DB_PATH)
        conn.execute('''CREATE TABLE preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL, response_a TEXT NOT NULL, response_b TEXT NOT NULL,
            winner TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        )''')
        conn.execute("INSERT INTO preferences (prompt, response_a, response_b, winner) VALUES ('p','a','b','a')")
        conn.commit()
        conn.close()
        self.assertEqual(preferences.count(), 1)
        self.assertEqual(preferences.list_recent(1)[0]['source'], 'human')


if __name__ == '__main__':
    unittest.main()
