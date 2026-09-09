"""Metadata preflight tests; not ANN correctness tests."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import h5py


class MetadataProbeTest(unittest.TestCase):
    def test_complete_small_file_metadata_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "small.h5"
            output = Path(tmp) / "metadata.json"
            with h5py.File(str(source), "w") as f:
                f.attrs["distance"] = "angular"
                f.create_dataset("train", shape=(3, 4), dtype="f4")
            cmd = [sys.executable, "scripts/phase5a_inspect_header.py",
                   str(source), "--artifact-size", str(source.stat().st_size),
                   "--output", str(output)]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
            result = json.loads(output.read_text())
            self.assertEqual(result["datasets"]["train"]["shape"], [3, 4])
            self.assertEqual(result["status"], "METADATA_ONLY_NOT_VALIDATED_DATASET")
            prior = output.read_bytes()
            retry = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual(output.read_bytes(), prior)


if __name__ == "__main__":
    unittest.main()
