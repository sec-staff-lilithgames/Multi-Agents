# Global Rules for All Agents

These rules are mandatory. Agents must implement them exactly and fail fast when violated.

## Execution & Isolation
- Use a dedicated Codex CLI session per agent process; never share venvs across projects.
- Honor `CODEX_SESSION_BASEDIR` (project/.camel_sessions) for all subprocesses to ensure isolation.
- MCP transport is `stdio` by default; do not invent ad‑hoc IPC.

## User Interaction
- Only `broker` interacts with the user. `boss` and `workers` are never user‑facing.
- Prefer clarification over guesswork: if intent/params are missing or ambiguous, request clarifications before executing.
- Never silently default to unrelated demos (e.g., C hello‑world) when requirements are unclear.

## Requirement Parsing
- Primary source is `requirement.md`. Supported sections (case-insensitive): `Intent`, `Task`, `Params`, `Environment`, `Outputs`, `Acceptance`, `Limits`.
- Broker must parse markdown and merge with clarifications (responses override file when in conflict).
- Broker must leverage LLM to analyze free-form requirements into structured MetaInfo. If the LLM is unavailable or errors, the broker must halt and report the unavailability and its reason to the user; no other fallback strategies are permitted.

## Data‑Driven Behavior (No Hardcoding)
- Maintain environment diagnostics in `configs/diagnostic_rules.json` (regex patterns → `needs[]` + `message`).
- Maintain script templates in `configs/script_templates.json` or `configs/templates/*` (language‑specific) and render via placeholders. Do not embed large scripts in code.

## Environment Handling
- Workers must return structured errors for run/compile failures (`ok=false`, `returncode`, `stdout`, `stderr`, `error`).
- Boss must diagnose failures with data‑driven rules and respond using `action_required=environment` and a structured `needs[]` list and `message`.
- Broker must stop orchestration on `action_required`, persist `snapshot/needs.json`, surface to user, and resume only after user confirms.

## Projectization & Outputs
- Every run creates `project/<title-slug>-<hash>/` with:
  - `out/` — artifacts and `result.json`.
  - `snapshot/` — `snapshot.json` (roles, status hash, env), and optional `needs.json`.
  - `.camel_sessions/` — per‑project sessions for all roles.
  - A copy of `requirement.md`.
- Agents must write artifacts only under the current project’s `out/`.

## Public Attachments
- Users must place input attachments under `./attachments`.
- All agents may read and write in `./attachments` (or the absolute path in `ATTACHMENTS_DIR`).
- Treat attachments as shared inputs; final outputs still belong in each project’s `out/`.

## Role Boundaries
- Broker: parse+clarify; create project; set `CODEX_SESSION_BASEDIR`; call boss; copy artifacts; save snapshots.
- Boss: decide plan from broker `meta`; assign to worker1/worker2; no direct user I/O; perform diagnostics on failure.
- Worker1: file authoring, compilation, package install steps.
- Worker2: runtime execution and verification.

## Timeouts & Reliability
- MCP client operations should default to generous but bounded timeouts (≥120s for long steps); use clear error messages on timeout.
- Steps must be idempotent when re‑run after environment preparation.

## Security & Privacy
- Log minimal necessary details; never log secrets (API keys, tokens, cookies).
- External network access should be limited to what the task requires and follow the environment/network policy.

## Compliance
- If a rule conflicts with task code, the task code must be changed to conform. Keep guidelines and configs updated together.
