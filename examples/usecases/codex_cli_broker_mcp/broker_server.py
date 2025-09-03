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

    # -------- Markdown parsing helpers --------
    def _get_md_section(self, text: str, title: str) -> str:
        lines = text.splitlines()
        capture = False
        buf = []
        import re
        pat = re.compile(rf"^\s*#+\s*{re.escape(title)}\s*$", re.IGNORECASE)
        for i, ln in enumerate(lines):
            if pat.match(ln):
                capture = True
                continue
            if capture and ln.lstrip().startswith('#'):
                break
            if capture:
                buf.append(ln)
        return "\n".join(buf).strip()

    def _parse_params_list(self, body: str) -> Dict:
        params: Dict[str, str] = {}
        import re
        for ln in body.splitlines():
            m = re.match(r"^\s*[-*]\s*([A-Za-z0-9_\-]+)\s*:\s*(.+)\s*$", ln)
            if m:
                key, val = m.group(1), m.group(2)
                params[key] = val.strip()
        return params

    def _parse_requirement_md(self, text: str) -> Dict:
        """Parse requirement.md into structured fields if sections exist.

        Recognized headings (case-insensitive): Task, Intent, Params,
        Environment, Outputs, Acceptance, Limits.
        Returns dict with optional keys: intent, task_text, params, env,
        outputs, acceptance, limits.
        """
        d: Dict = {}
        task = self._get_md_section(text, "Task")
        if task:
            d["task_text"] = task
        intent = self._get_md_section(text, "Intent").strip()
        if intent:
            # take first non-empty line as intent
            first = next((ln.strip() for ln in intent.splitlines() if ln.strip()), "")
            if first:
                d["intent"] = first
        params_body = self._get_md_section(text, "Params")
        if params_body:
            d["params"] = self._parse_params_list(params_body)
        env = self._get_md_section(text, "Environment")
        if env:
            d["environment"] = env
        outputs = self._get_md_section(text, "Outputs")
        if outputs:
            d["outputs"] = outputs
        acceptance = self._get_md_section(text, "Acceptance")
        if acceptance:
            d["acceptance"] = acceptance
        limits = self._get_md_section(text, "Limits")
        if limits:
            d["limits"] = limits
        return d

    def _analyze_requirement(self, text: str) -> Dict:
        """Parse requirement via external rules (no LLM).

        Loads configs/intent_rules.json and applies regex matchers to decide
        intent/plan_type. Extractors populate params. Supports optional
        requirement.md sections (Task/Intent/Params). Falls back to default.
        """
        import json, re, os
        full_text = (text or "").strip()
        md = self._parse_requirement_md(full_text)
        t = md.get("task_text") or full_text
        rules_path = Path(__file__).parents[3] / "configs" / "intent_rules.json"
        if not rules_path.exists():
            return {"intent": "create_helloworld_c", "plan_type": "c", "requirements": {"filename": "helloworld.c", "binary": "hello", "expected_output": "Hello, World!\n"}, "raw_text": full_text}
        data = json.loads(rules_path.read_text(encoding="utf-8"))
        rules = data.get("rules", [])

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

        # Try LLM analysis first if available
        try:
            meta_llm = self._llm_analyze_requirement(full_text, attachments)
            if attachments:
                meta_llm["attachments"] = attachments
                meta_llm["attachments_dir"] = attach_dir
            return meta_llm
        except Exception:
            pass

        # Lightweight NLP helpers
        def extract_kv_anywhere(s: str) -> Dict[str, str]:
            params: Dict[str, str] = {}
            for ln in s.splitlines():
                m = re.match(r"^\s*[-*]?\s*([A-Za-z0-9_\-]+)\s*[:：]\s*(.+?)\s*$", ln)
                if m:
                    k, v = m.group(1), m.group(2)
                    params[k.strip()] = v.strip()
            m = re.search(r"(输出|保存)(?:为|到)?\s*([\w\-./\\]+\.[A-Za-z0-9]{2,6})", s)
            if m:
                params.setdefault("output", m.group(2))
            return params

        def extract_json_blocks(s: str) -> Dict[str, str]:
            try:
                blk = re.findall(r"```json\s*([\s\S]*?)```", s, re.IGNORECASE)
                merged = {}
                for b in blk:
                    obj = json.loads(b)
                    if isinstance(obj, dict):
                        merged.update(obj)
                return merged
            except Exception:
                return {}

        def extract_params_by_intent(intent: str, s: str, base: Dict[str, str]) -> Dict[str, str]:
            p = dict(base)
            if intent == "create_docx_line":
                if "line" not in p or not p.get("line"):
                    m = re.search(r"写(?:一行|一句)([^\n，。!！?？]+)", s)
                    if m:
                        p["line"] = m.group(1).strip()
                    else:
                        q = re.search(r"[\"“](.*?)[\"”]", s)
                        if q:
                            p["line"] = q.group(1).strip()
                if not p.get("output"):
                    m2 = re.search(r"(\w+\.docx)", s, re.IGNORECASE)
                    if m2:
                        p["output"] = m2.group(1)
            if intent == "weibo_hot_to_docx":
                if "count" not in p:
                    m = re.search(r"前(\d+)条", s)
                    if m:
                        p["count"] = m.group(1)
                if not p.get("output"):
                    m2 = re.search(r"(\w+\.docx)", s, re.IGNORECASE)
                    if m2:
                        p["output"] = m2.group(1)
            if intent == "web_weather_to_excel":
                if "hours" not in p:
                    m = re.search(r"未来(\d+)个?小时", s)
                    if m:
                        p["hours"] = m.group(1)
                if "date" not in p:
                    m = re.search(r"(20\d{2}[./-]\d{2}[./-]\d{2})", s)
                    if m:
                        p["date"] = m.group(1)
                if "city" not in p:
                    for city in ["北京", "北京市", "上海", "广州", "深圳", "杭州", "南京"]:
                        if city in s:
                            p["city"] = city.replace("市", "")
                            break
                if not p.get("output"):
                    m2 = re.search(r"(\w+\.xlsx)", s, re.IGNORECASE)
                    if m2:
                        p["output"] = m2.group(1)
            return p

        kv_any = extract_kv_anywhere(t)
        json_blk = extract_json_blocks(t)
        kv_all = {**kv_any, **json_blk, **(md.get("params") or {})}

        # Prefer explicit Intent section if present
        if md.get("intent"):
            for r in rules:
                if r.get("intent") == md.get("intent"):
                    req = r.get("requirements", {}).copy()
                    params_final = extract_params_by_intent(r.get("intent"), t, kv_all)
                    # defaults per intent
                    if r.get("intent") == "create_docx_line":
                        params_final.setdefault("line", "Hello, world")
                        req.setdefault("outputs", [params_final.get("output", "output.docx")])
                    if r.get("intent") == "weibo_hot_to_docx":
                        params_final.setdefault("count", "10")
                        req.setdefault("outputs", [params_final.get("output", "hot.docx")])
                    if r.get("intent") == "web_weather_to_excel":
                        params_final.setdefault("city", "北京")
                        params_final.setdefault("hours", "5")
                        req.setdefault("outputs", [params_final.get("output", "weather.xlsx")])
                    req.setdefault("params", {}).update(params_final)
                    meta = {"intent": r.get("intent"), "plan_type": r.get("plan_type"), "requirements": req, "raw_text": full_text}
                    if attachments:
                        meta["attachments"] = attachments
                        meta["attachments_dir"] = attach_dir
                    return meta

        # Otherwise match intents by free text
        for r in rules:
            pats = r.get("match_any", [])
            if any(re.search(p, t, re.IGNORECASE) for p in pats):
                req = r.get("requirements", {}).copy()
                params_final = extract_params_by_intent(r.get("intent"), t, kv_all)
                # render template if any
                template_key = r.get("template_key")
                if template_key and r.get("plan_type") == "python":
                    tmpl_path = Path(__file__).parents[3] / "configs" / "script_templates.json"
                    tmpl = None
                    try:
                        tmpl_all = json.loads(tmpl_path.read_text(encoding="utf-8"))
                        tmpl = (tmpl_all.get("python", {}) or {}).get(template_key)
                    except Exception:
                        tmpl = None
                    if not tmpl:
                        alt = Path(__file__).parents[3] / "configs" / "templates" / "python" / f"{template_key}.py.tmpl"
                        if alt.exists():
                            tmpl = alt.read_text(encoding="utf-8")
                    if tmpl:
                        p = params_final
                        if template_key == "web_weather_to_excel":
                            script_content = (
                                tmpl
                                .replace("__CITY__", str(p.get("city", "Beijing")))
                                .replace("__HOURS__", str(p.get("hours", "5")))
                                .replace("__DATE__", str(p.get("date", "")))
                                .replace("__OUTPUT__", str(p.get("output", (req.get("outputs") or ["weather.xlsx"])[0])))
                            )
                        elif template_key == "weibo_hot_to_docx":
                            script_content = (
                                tmpl
                                .replace("__COUNT__", str(p.get("count", "10")))
                                .replace("__OUTPUT__", str(p.get("output", (req.get("outputs") or ["hot.docx"])[0])))
                            )
                        else:
                            script_content = tmpl.format(output=p.get("output", (req.get("outputs") or ["out.txt"])[0]))
                        req["script_content"] = script_content
                        if p.get("output"):
                            req["outputs"] = [p["output"]]
                # defaults per intent
                if r.get("intent") == "create_docx_line":
                    params_final.setdefault("line", "Hello, world")
                    req.setdefault("outputs", [params_final.get("output", "output.docx")])
                if r.get("intent") == "weibo_hot_to_docx":
                    params_final.setdefault("count", "10")
                    req.setdefault("outputs", [params_final.get("output", "hot.docx")])
                if r.get("intent") == "web_weather_to_excel":
                    params_final.setdefault("city", "北京")
                    params_final.setdefault("hours", "5")
                    req.setdefault("outputs", [params_final.get("output", "weather.xlsx")])
                req.setdefault("params", {}).update(params_final)
                meta = {"intent": r.get("intent"), "plan_type": r.get("plan_type"), "requirements": req, "raw_text": full_text}
                if attachments:
                    meta["attachments"] = attachments
                    meta["attachments_dir"] = attach_dir
                return meta

        # If still unknown, return clarification rather than unrelated fallback
        supported = [r.get("intent") for r in rules if r.get("intent")]
        questions = [
            "未能识别意图。请用自然语言明确你的目标、期望的产物与约束（例如输出文件名/格式等）。",
            "若涉及具体工具/平台（如 Frida/ADB/C 编译），请在文本中直说。",
        ]
        return {
            "intent": None,
            "plan_type": None,
            "requirements": {},
            "raw_text": full_text,
            "clarification": {
                "action_required": "clarification",
                "missing": ["Intent"],
                "questions": questions,
                "template": "",
            },
            "attachments": attachments,
            "attachments_dir": attach_dir,
        }

    async def start(self, requirements_path: str, output_dir: Optional[str] = None, project_dir: Optional[str] = None, project_id: Optional[str] = None) -> Dict:
        try:
            req_path = Path(requirements_path)
            if not req_path.exists():
                return {"ok": False, "error": f"requirements file not found: {req_path}"}
            text = req_path.read_text(encoding="utf-8")
            meta = self._analyze_requirement(text)

            # If broker needs clarification, persist and return early
            if meta.get("clarification"):
                out_dir = Path(output_dir) if output_dir else (Path.cwd() / "out")
                out_dir.mkdir(parents=True, exist_ok=True)
                proj_dir = Path(project_dir) if project_dir else out_dir.parent
                snapshot_dir = proj_dir / "snapshot"
                snapshot_dir.mkdir(parents=True, exist_ok=True)

                clar = meta["clarification"]
                needs_doc = {
                    "action_required": "clarification",
                    "message": "需要补充必要参数以继续执行",
                    "questions": clar.get("questions", []),
                    "missing": clar.get("missing", []),
                    "template": clar.get("template", "")
                }
                (snapshot_dir / "needs.json").write_text(json.dumps(needs_doc, indent=2), encoding="utf-8")
                final_meta = {
                    "artifact": [],
                    "result": {"stdout": "", "stderr": "", "kind": None},
                    "requirements_file": str(req_path.resolve()),
                    "project": {"dir": str(proj_dir.resolve()), "id": project_id, "status_hash": ""},
                    "needs": needs_doc,
                }
                out_meta = out_dir / "result.json"
                out_meta.write_text(json.dumps(final_meta, indent=2), encoding="utf-8")
                return {"ok": False, "artifacts": [], "result_json": str(out_meta), "error": needs_doc["message"], "needs": needs_doc}

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
