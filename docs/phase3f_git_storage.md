# Phase 3F lossless Git transport

The five rotated database arrays are512,000,000 bytes each, above GitHub's
100MiB file limit. Their complete content is tracked as ordered48MiB chunks
under `runs/phase3f_rotation_stability_v1/git_storage/`, with SHA256 checksums
for every chunk and complete file. Original local arrays are retained and
ignored by Git; no raw result is deleted or resampled. The original
`final_verification.json` and all scientific outputs remain unchanged.

After cloning, reconstruct the five original paths before replaying the
experiment's original verification commands:

```bash
python3 scripts/phase3f_storage.py restore
python3 scripts/phase3f_storage.py verify
```

Restore refuses to overwrite mismatching existing files and verifies the
ordered byte stream before writing. No training or graph search is involved.
The source hashes match the historical Phase3F artifact audit.

Packaging command: `python3 scripts/phase3f_storage.py pack`.
To stay below the2GiB per-push limit, sync three consecutive additive commits:
scientific code/results/models; exhaustive-validation distances plus R0/R1
array chunks; then R2/R3/R4 array chunks. No history rewriting or force push.

Limits: [GitHub large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github),
[GitHub push limit](https://docs.github.com/en/get-started/using-git/troubleshooting-the-2-gb-push-limit).
