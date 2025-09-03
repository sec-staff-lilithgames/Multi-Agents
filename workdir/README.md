# Agent Private Work Directories

This folder contains per-role working directories. Each agent must use its own subdirectory.

- workdir/broker — readable/writable by broker and boss; not accessible to workers.
- workdir/boss — readable/writable by broker and all workers.
- workdir/worker1 — readable/writable by boss and other workers; not accessible to broker.
- workdir/worker2 — readable/writable by boss and other workers; not accessible to broker.

Note: On a single-user OS environment, strict cross-role file permissions may
not be enforceable. Agents simulate isolation by using separate subdirectories
and setting restrictive defaults (0700). Orchestration enforces policy.
