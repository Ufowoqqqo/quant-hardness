# Read-only host check before primary benchmark

After all stress/tests finished, `ps -eo pid,comm,pcpu --sort=-pcpu | head -12`
in the host-visible context showed codex0.7%, another codex0.2%, sshd-session
0.1%, systemd/kernel workers0.0%. No heavy competing job was observed.
No unrelated jobs were stopped, and no host frequency/turbo settings changed.

The runner's per-phase process snapshots are sandbox-namespace-local, not a
continuous host-wide monitor. Per-cell /proc/loadavg, /proc/stat, /proc/meminfo
and frequency snapshots are additionally preserved. This is a shared-host
experiment, not guaranteed exclusive machine isolation.
