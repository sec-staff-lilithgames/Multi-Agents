# Contracts: Messages and MCP Conventions

All agent communication must use the following shapes.

## Meta (Broker → Boss)
```
{
  "intent": "<string>",
  "plan_type": "bash" | "python" | "c" | null,
  "requirements": { ... },
  "raw_text": "<original requirement text>",
  // Optional when missing information
  "clarification"?: {
    "action_required": "clarification",
    "missing": ["ParamName", ...] | ["Intent"],
    "questions": ["..."],
    "template": "..."
  }
}
```

## Boss/Worker Failure Diagnosis → Broker
```
{
  "ok": false,
  "error": "<short label>",
  "action_required": "environment",
  "needs": [
    {"type": "binary"|"pip"|"device", "name"?: "<name>", "package"?: "<pkg>", "status": "<status>"}
  ],
  "message": "<actionable hint>",
  "stdout": "...",
  "stderr": "...",
  // Optional for snapshot
  "boss_session_dir"?: "...",
  "w1_session_dir"?: "...",
  "w2_session_dir"?: "..."
}
```

## Success Result (Boss → Broker)
```
{
  "ok": true,
  "kind": "<intent or branch>",
  "artifacts": ["/abs/path/to/artifact", ...],
  "stdout": "...",
  "stderr": "...",
  "boss_session_dir": "...",
  "w1_session_dir": "...",
  "w2_session_dir": "..."
}
```

## MCP Tool Exposure
- Workers expose a single tool `do_step(step: str, payload: dict, timeout?: float)` with steps: `write_file`, `compile`, `run`, `verify_output`.
- Boss exposes `execute(meta: dict)`.
- Broker (MCP server mode) exposes `start(requirements_path, output_dir, project_dir, project_id)`.

All tools must return `content[0].type == "text"` with a JSON string body following the contracts above.

