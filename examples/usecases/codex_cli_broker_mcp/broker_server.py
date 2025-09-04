import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, List

from camel.runtime.codex_cli_session import CodexCliSession
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.utils.mcp import MCPServer
from camel.utils.mcp_client import MCPClient


def spawn_server(script_path: Path):
    import sys, os

    python = sys.executable
    cmd = [python, str(script_path), "--mode", "stdio"]
    return {
        "command": cmd[0],
        "args": cmd[1:],
        "env": os.environ.copy(),
    }


@MCPServer(function_names=["start"], server_name="BrokerAgent")
class Broker:
    """Broker agent: reads user requirement file, derives basic meta, calls
    Boss agent, and persists artifacts to the provided output directory.

    Tool:
    - start: payload {"requirements_path": str, "output_dir": str}
      Returns: {"ok": bool, "artifacts": [str], "result_json": str}
    """

    def __init__(self) -> None:
        self.session = CodexCliSession.create()
        # Optional LLM assistant (API key set later). Safe to be None.
        self.llm = self._create_llm()

    # -------- LLM helpers --------
    def _create_llm(self) -> Optional[ChatAgent]:
        try:
            import os, json as _json
            
            # Force Qwen Plus as the only provider
            api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
            if not api_key:
                ks = Path.cwd() / "keystone"
                qp = ks / "qwen-plus.key"
                try:
                    if qp.exists():
                        content = qp.read_text(encoding="utf-8").strip()
                        api_key = (content.splitlines()[0] if content else None)
                except Exception:
                    pass
            
            if not api_key:
                return None
                
            model = ModelFactory.create(model_platform=ModelPlatformType.QWEN, model_type="qwen-plus", api_key=api_key)
            
            # Apply global guideline
            guideline = self._load_global_guideline()
            sys_prompt = (
                "[GLOBAL_GUIDELINE]\n" + guideline + "\n\n" if guideline else ""
            ) + "You are Broker agent. Follow all global guidelines strictly."
            
            return ChatAgent(system_message=sys_prompt, model=model)
        except Exception:
            return None
    
    def _load_global_guideline(self) -> str:
        try:
            g = Path(__file__).parents[3] / "guideline" / "global_rules.md"
            if g.exists():
                return g.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        return ""

    def _llm_analyze_requirement(self, text: str, attachments: List[Dict] | None = None) -> Dict:
        if not self.llm:
            raise RuntimeError("LLM not available")
        import json as _json, re as _re
        guideline = self._load_global_guideline()
        sys_prompt = (
            "[GLOBAL_GUIDELINE]\n" + guideline + "\n\n" if guideline else ""
        ) + (
            "You are a planning assistant. Convert the user's free-form requirement\n"
            "into a single JSON object (no prose) with keys: intent, plan_type,\n"
            "requirements. The requirements object should include for python: {script_name, script_content or pip+template params, outputs[]}\n"
            "for bash: {script_name, script_content, outputs[]} ; for c: {filename, content, binary, expected_output?}.\n"
            "Prefer precise, minimal plans that will run in a sandbox. Do not include explanations.\n"
        )
        att_section = ""
        if attachments:
            att_lines = [f"- {a.get('name')} => {a.get('path')}" for a in attachments]
            att_section = "\nAttached files:\n" + "\n".join(att_lines)
        user_msg = f"Requirement:\n{text}\n{att_section}\nOutput JSON only."
        resp = self.llm.step(user_msg)
        if resp.terminated or not resp.msgs:
            raise RuntimeError("LLM returned no content")
        content = resp.msgs[0].content.strip()
        m = _re.search(r"\{[\s\S]*\}\s*$", content)
        json_text = m.group(0) if m else content
        meta = _json.loads(json_text)
        if not isinstance(meta, dict):
            raise RuntimeError("LLM meta is not an object")
        meta.setdefault("raw_text", text)
        return meta

    def _analyze_requirement(self, text: str) -> Dict:
        """Parse requirement using the LLM only.

        Any failure here propagates as an exception so the caller can report
        the unavailability of the LLM and the underlying reason to the user.
        """
        import os
        full_text = (text or "").strip()

        # Scan attachments directory (public shared inputs)
        attachments = []
        attach_dir = os.environ.get("ATTACHMENTS_DIR") or str((Path.cwd() / "attachments").resolve())
        try:
            base = Path(attach_dir)
            if base.exists() and base.is_dir():
                for p in base.glob("**/*"):
                    if p.is_file() and not p.name.startswith("."):
                        try:
                            import mimetypes
                            mime, _ = mimetypes.guess_type(str(p))
                        except Exception:
                            mime = None
                        stat = p.stat()
                        attachments.append({
                            "name": p.name,
                            "path": str(p.resolve()),
                            "size": stat.st_size,
                            "mime": mime,
                            "rel": str(p.relative_to(base)) if str(p).startswith(str(base)) else p.name,
                        })
        except Exception:
            # best-effort scanning; do not fail analysis
            pass

        if not self.llm:
            raise RuntimeError("missing QWEN_API_KEY")

        meta_llm = self._llm_analyze_requirement(full_text, attachments)
        if attachments:
            meta_llm["attachments"] = attachments
            meta_llm["attachments_dir"] = attach_dir
        return meta_llm

    async def start(self, requirements_path: str, output_dir: Optional[str] = None, project_dir: Optional[str] = None, project_id: Optional[str] = None) -> Dict:
        try:
            req_path = Path(requirements_path)
            if not req_path.exists():
                return {"ok": False, "error": f"requirements file not found: {req_path}"}
            text = req_path.read_text(encoding="utf-8")
            try:
                meta = self._analyze_requirement(text)
            except Exception as e:
                return {"ok": False, "error": "LLM not available", "reason": str(e)}

            # Spawn boss server
            boss_script = Path(__file__).parents[1] / "codex_cli_boss_mcp" / "boss_server.py"
            if not boss_script.exists():
                return {"ok": False, "error": f"boss server not found: {boss_script}"}

            boss_cfg = spawn_server(boss_script)
            out_dir = Path(output_dir) if output_dir else (Path.cwd() / "out")
            out_dir.mkdir(parents=True, exist_ok=True)
            proj_dir = Path(project_dir) if project_dir else out_dir.parent
            snapshot_dir = proj_dir / "snapshot"
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            # Call boss
            async with MCPClient(boss_cfg) as boss:
                res_raw = await boss.call_tool("execute", {"meta": meta})
                content = res_raw.content[0]
                result = json.loads(content.text)

            # If boss requires environment/user input, persist needs and return early
            if not result.get("ok") and result.get("action_required") == "environment":
                needs_path = snapshot_dir / "needs.json"
                needs_doc = {
                    "message": result.get("message"),
                    "needs": result.get("needs", []),
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                }
                needs_path.write_text(json.dumps(needs_doc, indent=2), encoding="utf-8")

                final_meta = {
                    "artifact": [],
                    "result": {"stdout": result.get("stdout", ""), "stderr": result.get("stderr", ""), "kind": result.get("kind")},
                    "requirements_file": str(req_path.resolve()),
                    "project": {"dir": str(proj_dir.resolve()), "id": project_id, "status_hash": ""},
                    "needs": needs_doc,
                }
                out_meta = out_dir / "result.json"
                out_meta.write_text(json.dumps(final_meta, indent=2), encoding="utf-8")
                return {"ok": False, "artifacts": [], "result_json": str(out_meta), "error": result.get("message"), "needs": needs_doc}

            # Copy artifacts to project out dir
            artifacts = []
            for p in (result.get("artifacts", []) or []):
                src = Path(p)
                if src.exists():
                    dst = out_dir / src.name
                    shutil.copy2(src, dst)
                    artifacts.append(str(dst))

            # Build snapshot
            import hashlib, platform, sys as _sys, datetime as _dt
            status_seed = (text + (result.get("kind") or "") + "|" + "|".join(result.get("artifacts", []))).encode("utf-8")
            status_hash = hashlib.sha256(status_seed).hexdigest()

            roles = {
                "broker": {"session_dir": str(self.session.work_dir)},
                "boss": {"session_dir": result.get("boss_session_dir")},
                "worker1": {"session_dir": result.get("w1_session_dir")},
                "worker2": {"session_dir": result.get("w2_session_dir")},
            }

            snapshot = {
                "project_id": project_id or "unknown",
                "created_at": _dt.datetime.now().isoformat(),
                "requirements_file": str(req_path.resolve()),
                "summary": {"requirement_head": text.strip()[:140], "kind": result.get("kind")},
                "roles": roles,
                "out_dir": str(out_dir.resolve()),
                "status_hash": status_hash,
                "env": {
                    "python": _sys.version,
                    "platform": platform.platform(),
                    "CODEX_SESSION_BASEDIR": str(self.session.root_dir),
                },
                "attachments": meta.get("attachments", []),
                "attachments_dir": meta.get("attachments_dir"),
            }
            (snapshot_dir / "snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

            final_meta = {
                "artifact": artifacts,
                "result": {"stdout": result.get("stdout", ""), "stderr": result.get("stderr", ""), "kind": result.get("kind")},
                "requirements_file": str(req_path.resolve()),
                "project": {"dir": str(proj_dir.resolve()), "id": project_id, "status_hash": status_hash},
            }
            out_meta = out_dir / "result.json"
            out_meta.write_text(json.dumps(final_meta, indent=2), encoding="utf-8")

            return {"ok": bool(result.get("ok")), "artifacts": artifacts, "result_json": str(out_meta), "error": None if result.get("ok") else f"boss error: {result}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="stdio", choices=["stdio", "sse", "streamable-http"], help="MCP transport mode")
    args = parser.parse_args()

    worker = Broker()
    print(json.dumps({"session_dir": str(worker.session.work_dir)}), flush=True)
    worker.mcp.run(args.mode)


if __name__ == "__main__":
    main()
