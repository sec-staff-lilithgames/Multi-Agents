# Attachments (Public RW)

This directory is the shared attachment area for user-provided files.

- Purpose: Users place input files (datasets, images, docs, zips) here to be
  accessible by all agents.
- Visibility: Public within this repository runtime. Any agent may read and
  write files in this directory (RW). Agents should write final outputs to the
  current project's `out/` folder instead.
- Environment: Agents read the absolute path from `ATTACHMENTS_DIR`.
  If not set, default to `./attachments`.

Usage examples
- Place `spec.pdf` at `attachments/spec.pdf` and reference it in requirement.md.
- All agents can read/write attachments via `ATTACHMENTS_DIR` or `./attachments`.

Permissions
- Recommended permission: 0777 (public read/write) to allow multi-process and
  multi-role access.
