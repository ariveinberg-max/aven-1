import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from run_lock import WriterLock, WriterBusy, writer_active


class WriterLockTests(unittest.TestCase):
    def test_competing_process_and_crash_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            child = subprocess.Popen([sys.executable, '-u', '-c',
                'import sys,time; from run_lock import WriterLock; '
                'lock=WriterLock(sys.argv[1]); lock.__enter__(); print("locked",flush=True); time.sleep(30)', directory],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), 'locked')
                self.assertTrue(writer_active(directory))
                with self.assertRaises(WriterBusy):
                    with WriterLock(directory):
                        self.fail('Concurrent writer acquired lock')
            finally:
                child.kill()
                child.communicate(timeout=5)
            self.assertFalse(writer_active(directory))
            with WriterLock(directory):
                self.assertTrue(writer_active(directory))
            self.assertFalse(writer_active(directory))

    def test_probe_has_no_side_effects_and_reuse_preserves_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new'
            self.assertFalse(writer_active(output))
            self.assertFalse(output.exists())
            with WriterLock(output):
                pass
            path = output / '.writer.lock'
            before = (path.read_bytes(), path.stat().st_mtime_ns)
            with WriterLock(output):
                pass
            self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
