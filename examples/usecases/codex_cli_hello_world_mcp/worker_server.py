import argparse
import json
import os
from typing import Dict, Optional

from camel.runtime.codex_cli_session import CodexCliSession
from camel.utils.mcp import MCPServer
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType


@MCPServer(function_names=["do_step"], server_name="WorkerAgent")
class Worker:
    """A simple worker exposing a single MCP tool `do_step`.

    Supports steps:
    - write_file: payload {"filename": str, "content": str}
    - compile: payload {"filename": str, "output": str}
    - run: payload {"cmd": [str, ...]}  # executed in isolated session
    - verify_output: payload {"expected": str, "from_run": str}
    """

    def __init__(self, role: str):
        self.role = role
        self.debug = bool(os.environ.get("CODEX_DEBUG"))
        # Ensure private role workdir with strict permissions
        from pathlib import Path as _P
        import os as _os
        role_map = {
            "writer_compiler": "worker1",
            "runner_verifier": "worker2",
        }
        priv = _P.cwd() / "workdir" / role_map.get(role, role)
        priv.mkdir(parents=True, exist_ok=True)
        try:
            _os.chmod(priv, 0o700)
        except Exception:
            pass
        # Load .codex env if available
        try:
            import json as _json
            cfg = _P.cwd() / ".codex" / "config.json"
            if cfg.exists():
                data = _json.loads(cfg.read_text(encoding="utf-8"))
                for k, v in (data.get("env", {}) or {}).items():
                    _os.environ.setdefault(str(k), str(v))
        except Exception:
            pass
        # Attachments dir fallback
        if not _os.environ.get("ATTACHMENTS_DIR"):
            pub = _P.cwd() / "attachments"
            pub.mkdir(parents=True, exist_ok=True)
            try:
                _os.chmod(pub, 0o777)
            except Exception:
                pass
            _os.environ["ATTACHMENTS_DIR"] = str(pub.resolve())
        self.session = CodexCliSession.create()
        # Optional LLM assistant for quick diagnostics
        self.llm = self._create_llm()

    def _create_llm(self) -> Optional[ChatAgent]:
        try:
            import os, json as _json
            from pathlib import Path as _P
            
            # Force Qwen Plus as the only provider
            api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
            if not api_key:
                ks = _P.cwd() / "keystone"
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
            ) + f"You are Worker agent with role: {self.role}. Follow all global guidelines strictly."
            
            return ChatAgent(system_message=sys_prompt, model=model)
        except Exception:
            return None
    
    def _load_global_guideline(self) -> str:
        try:
            from pathlib import Path as _P
            g = _P(__file__).parents[3] / "guideline" / "global_rules.md"
            if g.exists():
                return g.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        return ""

    def _llm_hint(self, stdout: str, stderr: str) -> Optional[str]:
        if not self.llm:
            return None
        try:
            guideline = self._load_global_guideline()
            prompt = (
                "[GLOBAL_GUIDELINE]\n" + guideline + "\n\n" if guideline else ""
            ) + (
                "Given stdout/stderr of a failed step, provide a single Chinese sentence hint\n"
                "on likely root cause or next action. Keep it under 30 characters."
            )
            msg = f"STDOUT:\n{(stdout or '').strip()}\n\nSTDERR:\n{(stderr or '').strip()}\n"
            resp = self.llm.step(msg)
            return (resp.msgs[0].content.strip() if resp.msgs else None) or None
        except Exception:
            return None

    # ----------------- Auto-repair helpers -----------------
    def _extract_missing_module(self, stdout: str, stderr: str) -> Optional[str]:
        import re
        s = f"{stdout}\n{stderr}"
        m = re.search(r"(?:ModuleNotFoundError|ImportError):\s+No module named ['\"]([A-Za-z0-9_\.]+)['\"]", s)
        if m:
            mod = m.group(1)
            return mod.split('.')[0]
        return None

    def _map_module_to_pip(self, module: str) -> str:
        mapping = {
            "docx": "python-docx",
            "PIL": "pillow",
            "yaml": "pyyaml",
            "cv2": "opencv-python",
            "sklearn": "scikit-learn",
            "bs4": "beautifulsoup4",
            "Crypto": "pycryptodome",
        }
        return mapping.get(module, module)

    def _llm_suggest_packages(self, stdout: str, stderr: str) -> list[str]:
        if not self.llm:
            return []
        try:
            guideline = self._load_global_guideline()
            prompt = (
                "[GLOBAL_GUIDELINE]\n" + guideline + "\n\n" if guideline else ""
            ) + (
                "你是Python依赖修复助手。根据下面的stdout/stderr判断缺失的第三方库，"
                "仅输出JSON: {\"packages\":[\"pkg\",...]}，不要输出其他内容。"
            )
            msg = f"STDOUT:\n{(stdout or '').strip()}\n\nSTDERR:\n{(stderr or '').strip()}\n"
            resp = self.llm.step(msg)
            import json as _json, re as _re
            content = resp.msgs[0].content.strip() if resp.msgs else "{}"
            m = _re.search(r"\{[\s\S]*\}\s*$", content)
            data = _json.loads(m.group(0) if m else content)
            pkgs = data.get("packages") if isinstance(data, dict) else []
            if isinstance(pkgs, list):
                return [str(p) for p in pkgs if p]
            return []
        except Exception:
            return []

    def _attempt_auto_repair_run(self, cmd: list[str], stdout: str, stderr: str, timeout: Optional[float]) -> Dict:
        # Heuristic: missing python module
        missing = self._extract_missing_module(stdout, stderr)
        candidates: list[str] = []
        if missing:
            candidates.append(self._map_module_to_pip(missing))
        else:
            candidates = self._llm_suggest_packages(stdout, stderr)

        attempted: list[str] = []
        if candidates:
            for pkg in candidates[:5]:
                attempted.append(pkg)
                try:
                    cp = self.session.run(["pip", "install", pkg], timeout=timeout)
                except Exception as e:
                    import subprocess as _sp
                    if isinstance(e, _sp.CalledProcessError):
                        return {
                            "ok": False,
                            "error": "pip install failed",
                            "stdout": e.stdout,
                            "stderr": e.stderr,
                            "action_required": "environment",
                            "needs": [{"type": "network", "name": "pypi", "status": "unreachable_or_blocked"}],
                            "attempted": attempted,
                        }
                    return {"ok": False, "error": str(e), "attempted": attempted}
            # Re-run original command after installs
            try:
                cp2 = self.session.run(cmd, timeout=timeout)
                return {"ok": True, "stdout": cp2.stdout, "stderr": cp2.stderr}
            except Exception as e:
                import subprocess as _sp
                if isinstance(e, _sp.CalledProcessError):
                    return {"ok": False, "error": str(e), "returncode": e.returncode, "stdout": e.stdout, "stderr": e.stderr, "attempted": attempted}
                return {"ok": False, "error": str(e), "attempted": attempted}

        # If no candidates and stderr mentions 'command not found', escalate
        s = f"{stdout}\n{stderr}".lower()
        if "command not found" in s:
            import re
            m = re.search(r"command not found: ([a-z0-9_\-]+)", s)
            name = m.group(1) if m else "unknown"
            return {"ok": False, "error": "missing binary", "action_required": "environment", "needs": [{"type": "binary", "name": name, "status": "missing"}]}

        # Unknown failure
        return {"ok": False, "error": "unhandled run failure", "stdout": stdout, "stderr": stderr}

    def do_step(self, step: str, payload: Dict, timeout: Optional[float] = 20.0) -> Dict:
        try:
            if self.debug:
                print(f"[DEBUG][worker:{self.role}] step={step} payload_keys={list(payload.keys())}", flush=True)
            if step == "write_file":
                filename = payload["filename"]
                content = payload["content"]
                fpath = self.session.work_dir / filename
                fpath.write_text(content)
                return {"ok": True, "path": str(fpath)}

            if step == "compile":
                filename = payload["filename"]
                output = payload.get("output", "a.out")
                cc = os.environ.get("CC", "cc")
                try:
                    cp = self.session.run([cc, filename, "-o", output], timeout=timeout)
                    return {
                        "ok": True,
                        "stdout": cp.stdout,
                        "stderr": cp.stderr,
                        "exe": str(self.session.work_dir / output),
                    }
                except Exception as e:
                    import subprocess as _sp
                    if isinstance(e, _sp.CalledProcessError):
                        s = (e.stdout or "") + "\n" + (e.stderr or "")
                        if "not found" in s.lower() or "command not found" in s.lower():
                            return {
                                "ok": False,
                                "error": str(e),
                                "returncode": e.returncode,
                                "stdout": e.stdout,
                                "stderr": e.stderr,
                                "action_required": "environment",
                                "needs": [{"type": "binary", "name": cc, "status": "missing_or_broken"}],
                                "llm_hint": self._llm_hint(e.stdout, e.stderr),
                            }
                        return {"ok": False, "error": str(e), "returncode": e.returncode, "stdout": e.stdout, "stderr": e.stderr, "llm_hint": self._llm_hint(e.stdout, e.stderr)}
                    return {"ok": False, "error": str(e)}

            if step == "run":
                cmd = payload["cmd"]
                try:
                    cp = self.session.run(cmd, timeout=timeout)
                    return {"ok": True, "stdout": cp.stdout, "stderr": cp.stderr}
                except Exception as e:
                    import subprocess as _sp
                    if isinstance(e, _sp.CalledProcessError):
                        repair = self._attempt_auto_repair_run(cmd, e.stdout or "", e.stderr or "", timeout)
                        if repair.get("ok"):
                            return repair
                        if repair.get("action_required") == "environment":
                            repair.setdefault("llm_hint", self._llm_hint(e.stdout, e.stderr))
                            return repair
                        return {"ok": False, "error": str(e), "returncode": e.returncode, "stdout": e.stdout, "stderr": e.stderr, "llm_hint": self._llm_hint(e.stdout, e.stderr), "repair": repair}
                    return {"ok": False, "error": str(e)}

            if step == "verify_output":
                expected = payload["expected"].strip()
                got = payload["from_run"].strip()
                return {"ok": got == expected, "expected": expected, "got": got}

            return {"ok": False, "error": f"unknown step {step}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, help="worker role name")
    parser.add_argument("--mode", default="stdio", choices=["stdio", "sse", "streamable-http"], help="MCP transport mode")
    args = parser.parse_args()

    worker = Worker(role=args.role)
    print(json.dumps({"session_dir": str(worker.session.work_dir)}))
    worker.mcp.run(args.mode)


if __name__ == "__main__":
    main()
