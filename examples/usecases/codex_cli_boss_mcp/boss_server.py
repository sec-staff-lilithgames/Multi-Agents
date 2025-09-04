# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Optional

from camel.runtime.codex_cli_session import CodexCliSession
from camel.utils.mcp import MCPServer
from camel.utils.mcp_client import MCPClient
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType


def log_progress(role: str, message: str) -> None:
    log_dir = os.environ.get("CODEX_LOG_DIR")
    if not log_dir:
        return
    try:
        p = Path(log_dir) / f"{role}.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(message.rstrip() + "\n")
    except Exception:
        pass

def spawn_worker(role: str, worker_script: Path):
    import sys

    python = sys.executable
    cmd = [python, str(worker_script), "--role", role, "--mode", "stdio"]
    return {
        "command": cmd[0],
        "args": cmd[1:],
        "env": os.environ.copy(),
    }


@MCPServer(function_names=["execute"], server_name="BossAgent")
class Boss:
    """Boss agent responsible for assigning work to two workers based on meta.

    Tool:
    - execute: payload {"meta": dict}
      Returns: {"ok": bool, "artifacts": [str], "kind": str, "stdout": str, "stderr": str}
    """

    def __init__(self, debug: bool = False):
        self.debug = bool(debug or os.environ.get("CODEX_DEBUG"))
        # Ensure private role workdir with strict permissions
        role_priv = Path.cwd() / "workdir" / "boss"
        role_priv.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(role_priv, 0o700)
        except Exception:
            pass
        # Load .codex env if available
        try:
            import json as _json
            cfg = Path.cwd() / ".codex" / "config.json"
            if cfg.exists():
                data = _json.loads(cfg.read_text(encoding="utf-8"))
                for k, v in (data.get("env", {}) or {}).items():
                    os.environ.setdefault(str(k), str(v))
        except Exception:
            pass
        # Attachments dir fallback
        if not os.environ.get("ATTACHMENTS_DIR"):
            pub = Path.cwd() / "attachments"
            pub.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(pub, 0o777)
            except Exception:
                pass
            os.environ["ATTACHMENTS_DIR"] = str(pub.resolve())
        self.session = CodexCliSession.create()
        # Optional LLM assistant for planning/diagnostics
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
            ) + "You are Boss agent. Follow all global guidelines strictly."
            
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

    async def _diagnose(self, worker_client: MCPClient, session_dir: Path, ctx: Dict) -> Dict:
        """Use LLM to diagnose common environment issues based on stdout/stderr.

        Returns a dict. If environment action required, include keys:
          action_required: "environment", message: str, needs: list
        Otherwise, return {action_required: "none", message: hint}
        """
        stdout = (ctx.get("stdout") or "").strip()
        stderr = (ctx.get("stderr") or "").strip()
        # Fallback heuristic if LLM not available
        def _fallback() -> Dict:
            s = f"{stdout}\n{stderr}".lower()
            needs = []
            message = ""
            if "pip" in s and ("ssl" in s or "proxy" in s or "connection" in s or "timeout" in s or "unreachable" in s):
                message = "pip 安装失败，可能是网络/证书/代理问题，请配置可用网络或预装依赖。"
                needs.append({"type": "network", "name": "pypi", "status": "unreachable_or_blocked"})
                return {"action_required": "environment", "needs": needs, "message": message}
            if "command not found" in s and ("cc" in s or "gcc" in s or "clang" in s):
                message = "缺少 C 编译器（cc/gcc/clang）。"
                needs.append({"type": "binary", "name": "cc/gcc/clang", "status": "missing"})
                return {"action_required": "environment", "needs": needs, "message": message}
            return {"action_required": "none", "message": "未能自动诊断，请查看 stderr"}

        if not self.llm:
            return _fallback()

        guideline = self._load_global_guideline()
        prompt = (
            "[GLOBAL_GUIDELINE]\n" + guideline + "\n\n" if guideline else ""
        ) + (
            "You are a build-and-run troubleshooter. Given stdout/stderr from a failed step,\n"
            "return a compact JSON object with keys: action_required ('environment'|'none'), message (Chinese),\n"
            "needs (list; each item like {type: 'pip'|'binary'|'network'|'device', name: string, status: string}).\n"
            "Be concise and only output JSON."
        )
        user = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n\nOutput JSON only."
        try:
            resp = self.llm.step(user)
            content = resp.msgs[0].content.strip() if resp.msgs else "{}"
            import re as _re, json as _json
            m = _re.search(r"\{[\s\S]*\}\s*$", content)
            data = _json.loads(m.group(0) if m else content)
            if not isinstance(data, dict):
                return _fallback()
            return data
        except Exception:
            return _fallback()

    def _decide_kind(self, meta: Dict) -> str:
        intent = (meta or {}).get("intent") or ""
        if intent == "get_android_build_fingerprint":
            return intent
        if intent == "create_docx_line":
            return intent
        return "create_helloworld_c"

    async def _run_workers(self, meta: Dict) -> Dict:
        if self.debug:
            msg = f"meta received: plan_type={meta.get('plan_type')} intent={meta.get('intent')}"
            print(f"[DEBUG][boss] {msg}", flush=True)
            log_progress("boss", msg)
        # Locate shared worker server
        worker_script = Path(__file__).parents[1] / "codex_cli_hello_world_mcp" / "worker_server.py"
        if not worker_script.exists():
            raise FileNotFoundError(f"worker server not found: {worker_script}")

        # Spawn two workers
        w1_cfg = spawn_worker("writer_compiler", worker_script)
        w2_cfg = spawn_worker("runner_verifier", worker_script)
        if self.debug:
            msg = "spawned worker configs"
            print(f"[DEBUG][boss] {msg}", flush=True)
            log_progress("boss", msg)

        kind = self._decide_kind(meta)
        if self.debug:
            msg = f"decided kind: {kind}"
            print(f"[DEBUG][boss] {msg}", flush=True)
            log_progress("boss", msg)

        async with MCPClient(w1_cfg, timeout=120.0) as w1, MCPClient(w2_cfg, timeout=120.0) as w2:
            from camel.utils.mcp_client import ServerConfig  # type: ignore

            def _parse(res):
                content_list = getattr(res, "content", None)
                content = content_list[0]
                return json.loads(getattr(content, "text", "{}"))

            # Branches
            # Generic Python plan if provided by broker
            if (meta.get("plan_type") == "python") and isinstance(meta.get("requirements"), dict) and meta["requirements"].get("script_content"):
                if self.debug:
                    msg = "branch: python-inline-script"
                    print(f"[DEBUG][boss] {msg}", flush=True)
                    log_progress("boss", msg)
                reqm = meta["requirements"]
                script_name = reqm.get("script_name", "main.py")
                script_content = reqm.get("script_content", "")
                outputs = reqm.get("outputs", []) or []
                pip_deps = reqm.get("pip", []) or []
                res1 = _parse(await w1.call_tool("do_step", {"step": "write_file", "payload": {"filename": script_name, "content": script_content}}))
                if not res1.get("ok"):
                    return {"ok": False, "error": f"write_file failed: {res1}"}
                script_path = Path(res1["path"]).resolve()
                session_dir = script_path.parent
                if pip_deps:
                    res2 = _parse(await w1.call_tool("do_step", {"step": "run", "payload": {"cmd": ["pip", "install", *pip_deps]}}))
                    if not res2.get("ok"):
                        diag = await self._diagnose(w1, session_dir, {"stdout": res2.get("stdout", ""), "stderr": res2.get("stderr", "")})
                        if diag.get("action_required") == "environment":
                            diag.update({"boss_session_dir": str(self.session.work_dir), "w1_session_dir": str(session_dir), "w2_session_dir": None, "error": "pip failed"})
                            return diag
                        return {"ok": False, "error": "pip failed"}
                res3 = _parse(await w1.call_tool("do_step", {"step": "run", "payload": {"cmd": ["python", script_name]}}))
                if not res3.get("ok"):
                    diag = await self._diagnose(w1, session_dir, {"stdout": res3.get("stdout", ""), "stderr": res3.get("stderr", "")})
                    if diag.get("action_required") == "environment":
                        diag.update({"boss_session_dir": str(self.session.work_dir), "w1_session_dir": str(session_dir), "w2_session_dir": None, "error": "script run failed"})
                        return diag
                    return {"ok": False, "error": "script run failed"}
                # Collect outputs
                artifacts = []
                for fn in outputs:
                    p = (session_dir / fn).resolve()
                    if p.exists():
                        artifacts.append(str(p))
                return {"ok": True, "kind": meta.get("intent") or "python", "artifacts": artifacts, "stdout": res3.get("stdout", ""), "stderr": res3.get("stderr", ""), "boss_session_dir": str(self.session.work_dir), "w1_session_dir": str(session_dir), "w2_session_dir": None}

            if kind == "create_docx_line":
                if self.debug:
                    msg = "branch: create_docx_line"
                    print(f"[DEBUG][boss] {msg}", flush=True)
                    log_progress("boss", msg)
                line = ((meta.get("requirements") or {}).get("params") or {}).get("line") or "Hello, world"
                script_name = "make_docx.py"
                docx_name = "output.docx"
                script_body = (
                    "from docx import Document\n"
                    "doc = Document()\n"
                    f"doc.add_paragraph({line!r})\n"
                    f"doc.save({docx_name!r})\n"
                )

                res1 = _parse(await w1.call_tool("do_step", {"step": "write_file", "payload": {"filename": script_name, "content": script_body}}))
                if not res1.get("ok"):
                    return {"ok": False, "error": f"write_file failed: {res1}"}
                script_path = Path(res1["path"]).resolve()
                session_dir = script_path.parent

                res2 = _parse(await w1.call_tool("do_step", {"step": "run", "payload": {"cmd": ["pip", "install", "python-docx"]}}))
                if not res2.get("ok"):
                    return {
                        "ok": False,
                        "error": "pip failed",
                        "action_required": "environment",
                        "needs": [{"type": "pip", "package": "python-docx", "status": "install_failed"}],
                        "message": "依赖安装失败，请确保网络可用或预先安装 python-docx。",
                        "stdout": res2.get("stdout", ""),
                        "stderr": res2.get("stderr", ""),
                        "boss_session_dir": str(self.session.work_dir),
                        "w1_session_dir": str(session_dir),
                        "w2_session_dir": None,
                    }

                res3 = _parse(await w1.call_tool("do_step", {"step": "run", "payload": {"cmd": ["python", script_name]}}))
                if not res3.get("ok"):
                    return {"ok": False, "error": f"script run failed: {res3}"}

                produced = (session_dir / docx_name).resolve()
                return {
                    "ok": True,
                    "kind": kind,
                    "artifacts": [str(produced)],
                    "stdout": res3.get("stdout", ""),
                    "stderr": res3.get("stderr", ""),
                    "boss_session_dir": str(self.session.work_dir),
                    "w1_session_dir": str(session_dir),
                    "w2_session_dir": None,
                }

            if kind == "get_android_build_fingerprint":
                if self.debug:
                    msg = "branch: get_android_build_fingerprint"
                    print(f"[DEBUG][boss] {msg}", flush=True)
                    log_progress("boss", msg)
                script_name = "get_fingerprint.sh"
                out_name = "fingerprint.txt"
                script = f"""
#!/usr/bin/env bash
set -euo pipefail

if ! command -v adb >/dev/null 2>&1; then
  echo "ERROR: adb not found in PATH" >&2
  exit 1
fi

serial=$(adb devices | awk '$2=="device" {{print $1; exit}}')
if [ -z "${{serial:-}}" ]; then
  echo "ERROR: no connected device in 'device' state" >&2
  adb devices -l || true
  exit 2
fi

fp=$(adb -s "$serial" shell getprop ro.build.fingerprint | tr -d '\r')
echo "$fp" > {out_name}
echo "FINGERPRINT: $fp"
""".lstrip()

                res1 = _parse(await w1.call_tool("do_step", {"step": "write_file", "payload": {"filename": script_name, "content": script}}))
                if not res1.get("ok"):
                    return {"ok": False, "error": f"write_file failed: {res1}"}
                script_path = Path(res1["path"]).resolve()
                session_dir = script_path.parent

                res2 = _parse(await w1.call_tool("do_step", {"step": "run", "payload": {"cmd": ["bash", script_name]}}))
                if not res2.get("ok"):
                    rc = res2.get("returncode")
                    stderr = res2.get("stderr", "") or ""
                    needs = []
                    msg = ""
                    if rc == 1 or "adb not found" in stderr.lower():
                        needs.append({"type": "binary", "name": "adb", "status": "missing"})
                        msg = "需要安装 Android adb 并将其加入 PATH。"
                    elif rc == 2 or "no connected device" in stderr.lower():
                        needs.append({"type": "device", "name": "android", "status": "not_connected_or_unauthorized"})
                        msg = "需要连接并授权至少一台处于 device 状态的 Android 设备。"
                    else:
                        msg = "脚本执行失败，请检查 adb 与设备连接。"
                    return {
                        "ok": False,
                        "error": "script run failed",
                        "action_required": "environment",
                        "needs": needs,
                        "message": msg,
                        "stdout": res2.get("stdout", ""),
                        "stderr": stderr,
                        "boss_session_dir": str(self.session.work_dir),
                        "w1_session_dir": str(session_dir),
                        "w2_session_dir": None,
                    }

                produced = (session_dir / out_name).resolve()
                return {
                    "ok": True,
                    "kind": kind,
                    "artifacts": [str(script_path), str(produced)] if produced.exists() else [str(script_path)],
                    "stdout": res2.get("stdout", ""),
                    "stderr": res2.get("stderr", ""),
                    "boss_session_dir": str(self.session.work_dir),
                    "w1_session_dir": str(session_dir),
                    "w2_session_dir": None,
                }

            # Default C hello world
            if self.debug:
                msg = "branch: default_c_hello"
                print(f"[DEBUG][boss] {msg}", flush=True)
                log_progress("boss", msg)
            req = {
                "filename": "helloworld.c",
                "content": '#include <stdio.h>\nint main(){ printf("Hello, World!\\n"); return 0; }\n',
                "expected_output": "Hello, World!\n",
                "binary": "hello",
            }

            res1 = _parse(await w1.call_tool("do_step", {"step": "write_file", "payload": {"filename": req["filename"], "content": req["content"]}}))
            if not res1.get("ok"):
                return {"ok": False, "error": f"write_file failed: {res1}"}
            src_path = Path(res1["path"]).resolve()

            res2 = _parse(await w1.call_tool("do_step", {"step": "compile", "payload": {"filename": req["filename"], "output": req["binary"]}}))
            if not res2.get("ok"):
                needs = []
                stderr = res2.get("stderr", "") or ""
                if "not found" in (res2.get("error", "") or "").lower() or "not found" in stderr.lower():
                    needs.append({"type": "binary", "name": os.environ.get("CC", "cc"), "status": "missing_or_broken"})
                return {
                    "ok": False,
                    "error": "compile failed",
                    "action_required": "environment",
                    "needs": needs,
                    "message": "编译失败，请确认已安装 C 编译器（cc/gcc/clang）。",
                    "stdout": res2.get("stdout", ""),
                    "stderr": stderr,
                }
            exe_path = Path(res2.get("exe")).resolve()

            res3 = _parse(await w2.call_tool("do_step", {"step": "run", "payload": {"cmd": [str(exe_path)]}}))
            if not res3.get("ok"):
                return {"ok": False, "error": f"run failed: {res3}"}

            res4 = _parse(await w2.call_tool("do_step", {"step": "verify_output", "payload": {"expected": req["expected_output"], "from_run": res3.get("stdout", "")}}))
            if not res4.get("ok"):
                return {"ok": False, "error": f"verify failed: {res4}"}

            return {
                "ok": True,
                "kind": kind,
                "artifacts": [str(src_path), str(exe_path)],
                "stdout": res3.get("stdout", ""),
                "stderr": res3.get("stderr", ""),
                "boss_session_dir": str(self.session.work_dir),
                "w1_session_dir": str(src_path.parent),
                "w2_session_dir": None,
            }

    async def execute(self, meta: Dict, timeout: Optional[float] = 120.0) -> Dict:
        try:
            return await self._run_workers(meta)
        except Exception as e:
            return {"ok": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="stdio", choices=["stdio", "sse", "streamable-http"], help="MCP transport mode")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        os.environ["CODEX_DEBUG"] = "1"

    worker = Boss(debug=args.debug)
    print(json.dumps({"session_dir": str(worker.session.work_dir)}), flush=True)
    if worker.debug:
        log_progress("boss", f"session: {worker.session.work_dir}")
    worker.mcp.run(args.mode)


if __name__ == "__main__":
    main()
