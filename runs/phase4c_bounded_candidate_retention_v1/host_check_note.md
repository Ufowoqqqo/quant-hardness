# Host-process pre-benchmark check

After validation/ctest had completed and before launching the benchmark,
the read-only host-visible command
`ps -eo pid,comm,pcpu --sort=-pcpu | head -12` returned:

```
    PID COMMAND         %CPU
1061816 codex            0.7
   6885 codex            0.2
1120844 sshd-session     0.1
      1 systemd          0.0
      2 kthreadd         0.0
      3 pool_workqueue_  0.0
      4 kworker/R-rcu_g  0.0
      5 kworker/R-sync_  0.0
      6 kworker/0:0H-ev  0.0
      7 kworker/R-netns  0.0
      9 kworker/0:0H-ev  0.0
```

No competing heavy job was observed. This was not a continuous global process
monitor. Runner environment snapshots use the sandbox PID namespace and show
the runner itself, not a full host process list. Its initial81.5% Python CPU
reading is the runner's completed hash verification, before timed execution;
no concurrent agent-owned analysis/build/validation worker was launched during
the benchmark. Host load/frequency may still vary on this nonexclusive host.
No processes were killed and no governor/turbo/affinity settings outside the
benchmark's own taskset process were changed.
