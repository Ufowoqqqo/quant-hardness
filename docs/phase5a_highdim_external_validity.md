# Phase 5A — acquisition/preflight status, not experimental results

## 2026-09-09 amendment (current)

The user approved the locally cached Qdrant text-embedding-3-large1536 version
instead of ada-002, before any ANN results. See
[binding amendment](phase5a_dataset_amendment.md) for the990000/10000 split,
base-only PQ training and cast-first normalization. The source/role checksum
manifest is under `runs/phase5a_highdim_external_validity_v1/amendment/`.
The following acquisition narrative is historical; no further ada-002
download is required for the amended experiment. Scientific/system gates
remain unevaluated. This workload is held-out entity-to-entity NN.
The full26-shard local validation now passes:1000000 rows, dimension1536,
and frozen990000 base/10000 query/65536 base-only training IDs. Three split/
preprocessing unit tests and full-size deterministic ID regeneration pass.
No GT/PQ training/graph build/ANN evaluation has run for Phase5A.

## Historical 2026-09-08 preflight

Status on 2026-09-08: **not run; no validated complete dataset available**.
The design is preregistered in
[`preregistration.md`](../runs/phase5a_highdim_external_validity_v1/preregistration.md)
and [`configuration`](../configs/indexes/phase5a_highdim_external_validity.conf).
No scientific or systems gate has been evaluated. No previous run is changed.

## Sources inspected

- [Upstream DBPedia/OpenAI dataset](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M),
  revision `af9b8869cc2d8debbd254d77737865bb09a2067f`: source metadata describes
  one million 1536-dimensional text-embedding-ada-002 vectors. It exposes
  26 Parquet files and a train split, not a separate official query split.
- [ANN-Benchmarks generator](https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/datasets.py)
  defines a deterministic 10,000-query split for this source. Generator
  inspection is not proof of the downloaded artifact's counts.
- [Qdrant benchmark registry](https://github.com/qdrant/vector-db-benchmark/blob/master/datasets/datasets.json)
  names `dbpedia-openai-1M-1536-angular`, cosine, with
  [this GCS archive](https://storage.googleapis.com/ann-filtered-benchmark/datasets/dbpedia_openai_1M.tgz).
  HTTP metadata: 5,098,686,329 bytes, generation1699276546881421,
  MD5 `dc5fecd77592b669643a5e1ea0541887`. A full transfer failed after
  15,113 bytes with curl92 (HTTP/2 stream closed). The partial file remains
  preserved under `runs/.../source/`; it is not an experimental input.
- The original ANN-Benchmarks HDF5 URL returned404. A matching
  [HDF5 mirror object](https://storage.googleapis.com/ann-datasets/ann-benchmarks/dbpedia-openai-1000k-angular.hdf5)
  returns200: 6,160,008,192 bytes, generation1695799664928427,
  MD5 `02bc868fc7c77442ef8eaad14c93daf3`.
  A 1MiB range downloaded in12.4s (~85KB/s); this does not validate the full
  object. An HTTP/1.1 attempt failed with curl35 TLS EOF. Whole-file SHA-256,
  content checks and GT remain pending. Prefix-only HDF5 metadata inspection
  confirms `train=(990000,1536)`, `test=(10000,1536)`, FP32, angular. Neighbor
  and distance object metadata are outside the downloaded prefix. This is
  explicitly not validation of any vector data.
  A full HDF5 transfer probe received2,867,200 bytes in30s, then stopped at
  the deliberately configured timeout (curl28), rather than a server error.
  At that measured rate the6.16GB transfer would take about18h; this is an
  estimate, not proof the dataset is inaccessible. No download is left running.

## Frozen decisions and unresolved prerequisites

Primary PQ768×8 is selected arithmetically, not by recall: 1536×4=6144B
FP32, 768B codes, 8× compression, subvector dimension2. Standard FAISS permits
this layout; performance and training feasibility are not yet measured.
Its 768KiB ADC table/query must count toward online cost. The machine has
approximately27GiB available RAM, enough to attempt memory-conscious loading
of this scale, not a guarantee that an unbounded loader fits.

Reserve65,536 official base-split IDs exclusively for training and keep all
official queries. This explicitly reduces the benchmark base; all GT must
be recomputed. Counts in the preregistration are expectations, not verified
measurements. Freeze actual role-ID checksums before training. No test query
will select parameters. All scoring uses archived unit-normalized FP32 L2.

The former ef64-only guard needs a small, reference-tested generalization;
no search code has been modified yet. Do not interpret the preflight as
completion of loader, bounded-retention, concurrency or GT tests.

## Decision

Questions about discovery robustness, ranking dominance, L16 recovery,
Recall/QPS frontier and paper-readiness are **unanswered** for this dataset.
Neither Case A/B/C nor a scientific next-experiment recommendation is justified
without the requested primary measurements. Required next operational step:
obtain a complete checksum-verified high-dimensional artifact through a
working transfer or a supplied local path, then execute the frozen design.
