# Project Structure & Snapshot Requirements

Every execution must create an isolated project directory:

```
project/
  <title-slug>-<hash>/
    requirement.md                 # input copy
    out/
      <artifacts...>
      result.json                  # final outcome (see contracts)
    snapshot/
      snapshot.json                # roles, status hash, env
      needs.json                   # present only if action_required occurred
    .camel_sessions/               # per-project session root for all roles
```

## Naming
- `<title-slug>` is derived from the first Markdown H1 in requirement.md (or the first non-empty line).
- `<hash>` is the first 8 hex characters of SHA-256(requirement.md contents).

## Snapshot Requirements
- `snapshot.json` must include:
  - `project_id`, absolute `out_dir`, `requirements_file`.
  - `summary.requirement_head` (first ~140 chars) and `summary.kind`.
  - `roles.{broker,boss,worker1,worker2}.session_dir`.
  - `status_hash` (hash of requirement text + kind + artifact paths).
  - `env` block: python, platform, CODEX_SESSION_BASEDIR.
- `needs.json` is created when clarification/environment is required, with `message`, `needs[]`, and optional `stdout/stderr`.

