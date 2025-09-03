# Roles Roster

This directory defines a global roster of agent roles and their Codex CLI startup items. Each role has a subdirectory containing a `.codex` folder with a `launch.json` describing how to start that role's process.

Current roles:

- broker: Interactive Codex CLI that parses requirement.md, asks user for clarifications in natural language, and orchestrates via boss. Invisible to workers.
- boss: Central dispatcher that receives MetaInfo from broker, optimizes task plans and assigns work to workers; periodically syncs progress and re-optimizes. Invisible to the user.
- worker1: Endpoint worker (writer/compilation duties) that executes boss-assigned task lists strictly, without autonomy.
- worker2: Endpoint worker (runtime/verification duties) that executes boss-assigned task lists strictly, without autonomy.

Model defaults

- broker: qwen-plus (via keystone/qwen-plus.key)
- boss: qwen-plus (via keystone/qwen-plus.key)
- worker1/worker2: qwen-plus (via keystone/qwen-plus.key)

Notes

- All paths are relative to the repository root. The launcher should set `CODEX_SESSION_BASEDIR` per project to isolate sessions.
- `mode` is `stdio` by default for MCP processes.
