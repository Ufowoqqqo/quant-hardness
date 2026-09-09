#!/usr/bin/env python3
"""Inspect HDF5 metadata from a downloaded prefix, never vector contents.

The sparse temporary file is NOT a dataset and must never be used for search.
HDF5 stores the expected EOF in its superblock; extending to that size permits
metadata inspection when the metadata is in the prefix. Missing metadata must
fail, not be guessed. No output vectors or retrieval labels are read.
"""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import h5py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prefix", type=Path)
    p.add_argument("--artifact-size", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prefix = a.prefix.read_bytes()
    if not 0 < len(prefix) <= a.artifact_size:
        raise ValueError("invalid prefix/artifact size")
    with tempfile.NamedTemporaryFile(prefix="phase5a-metadata-only-", suffix=".h5") as scratch:
        scratch.write(prefix)
        scratch.truncate(a.artifact_size)
        scratch.flush()
        with h5py.File(scratch.name, "r") as f:
            result = {
                "status": "METADATA_ONLY_NOT_VALIDATED_DATASET",
                "downloaded_prefix_bytes": len(prefix),
                "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                "expected_artifact_bytes": a.artifact_size,
                "h5py_version": h5py.__version__,
                "attributes": {k: str(v) for k, v in f.attrs.items()},
                "datasets": {k: ({"shape": list(v.shape), "dtype": str(v.dtype)}
                                 if v is not None else
                                 {"status": "OBJECT_METADATA_NOT_IN_PREFIX"})
                             for k, v in f.items()},
            }
    with a.output.open("x") as out:
        json.dump(result, out, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
