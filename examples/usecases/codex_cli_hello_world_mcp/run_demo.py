import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict

from camel.utils.mcp_client import MCPClient


def parse_mcp_result(result) -> Dict:
    """Parse MCP CallToolResult into a Python dict.

    Expects text content containing JSON dict produced by our worker.
    """
    try:
        content_list = getattr(result, "content", None)
        if not content_list:
            raise ValueError("empty content")
        content = content_list[0]
        if getattr(content, "type", None) == "text":
            text = getattr(content, "text", "")
            return json.loads(text)
        raise ValueError(f"unsupported content type: {getattr(content, 'type', None)}")
    except Exception as e:
        raise RuntimeError(f"Failed to parse MCP result: {e}. Raw={result}")


def spawn_worker(role: str):
    """Create a subprocess running the worker MCP server over stdio.

    Returns the MCPClient configuration to connect via stdio.
    """
    python = sys.executable
    cmd = [python, str(Path(__file__).with_name("worker_server.py")), "--role", role, "--mode", "stdio"]
    return {
        "command": cmd[0],
        "args": cmd[1:],
        "env": os.environ.copy(),
    }


async def call_tool(client: MCPClient, name: str, args: Dict):
    # Directly call tool by name; MCPClient validates availability
    return await client.call_tool(name, args)


async def main():
    # User-facing agent: receives TODO list and converts to meta info
    todo = "Write a C program helloworld.c that prints 'Hello, World!'"
    meta = {
        "goal": "create_helloworld_c",
        "requirements": {
            "filename": "helloworld.c",
            "content": '#include <stdio.h>\nint main(){ printf("Hello, World!\\n"); return 0; }\n',
            "expected_output": "Hello, World!\n",
            "binary": "hello",
        },
    }

    # Central node: spin up two workers (stdio MCP servers)
    w1_cfg = spawn_worker("writer_compiler")
    w2_cfg = spawn_worker("runner_verifier")

    # Connect MCP clients
    async with MCPClient(w1_cfg) as w1, MCPClient(w2_cfg) as w2:
        # Step 1: assign write_file to worker1
        req = meta["requirements"]
        res1_raw = await call_tool(w1, "do_step", {"step": "write_file", "payload": {"filename": req["filename"], "content": req["content"]}})
        res1 = parse_mcp_result(res1_raw)
        assert res1.get("ok"), f"write_file failed: {res1}"

        # Step 2: compile via worker1
        res2_raw = await call_tool(w1, "do_step", {"step": "compile", "payload": {"filename": req["filename"], "output": req["binary"]}})
        res2 = parse_mcp_result(res2_raw)
        assert res2.get("ok"), f"compile failed: {res2}"
        exe_path = res2.get("exe")

        # Step 3: run via worker2
        # Use absolute path to the built binary from worker1's session
        res3_raw = await call_tool(w2, "do_step", {"step": "run", "payload": {"cmd": [exe_path]}})
        res3 = parse_mcp_result(res3_raw)
        assert res3.get("ok"), f"run failed: {res3}"

        # Step 4: verify via worker2
        res4_raw = await call_tool(w2, "do_step", {"step": "verify_output", "payload": {"expected": req["expected_output"], "from_run": res3.get("stdout", "")}})
        res4 = parse_mcp_result(res4_raw)
        assert res4.get("ok"), f"verify failed: {res4}"

        # Central node: prepare final MetaInfo for user-facing agent
        final_meta = {
            "artifact": {
                "source_file": req["filename"],
                "binary": exe_path,
            },
            "result": {
                "stdout": res3.get("stdout", ""),
                "stderr": res3.get("stderr", ""),
                "verified": True,
            },
        }

        print(json.dumps(final_meta, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
