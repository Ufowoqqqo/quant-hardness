# Prebenchmark resource check

After GT, export, independent validation and CTest completed, a host-level
read-only `ps -eo pid,comm,pcpu --sort=-pcpu | head -12` showed only codex
processes at 0.7%/0.2%, sshd-session at0.1%, and kernel/system processes at0.0%.
No heavy competing job was found; no unrelated process was stopped.
No agent-owned GT, build, validation or plotting process runs concurrently
with the benchmark. CPU2 affinity is set by taskset; governor remains
performance and turbo remains enabled (intel_pstate/no_turbo=0).
No frequency, governor or privileged machine setting was changed.
Load/frequency/memory snapshots before and after benchmarks are recorded in
benchmark_environment.json. This is a shared machine, not exclusive-host
isolation; repetition variability remains part of the report.
