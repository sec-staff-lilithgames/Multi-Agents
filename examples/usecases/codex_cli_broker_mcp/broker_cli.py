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
import sys
from pathlib import Path
from typing import Dict, List

from camel.runtime.codex_cli_session import CodexCliSession
from camel.utils.mcp_client import MCPClient


def log_progress(role: str, message: str) -> None:
    """Append a progress message to the role-specific log file."""
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

def extract_title(text: str) -> str:
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            return s.lstrip("#").strip() or "project"
        return s
    return "project"


def slugify(name: str) -> str:
    import re
    s = (name or "project").strip().lower()
    s = s.replace(os.sep, "-").replace("/", "-")
    s = re.sub(r"[^\w\-\u4e00-\u9fff]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "project"


def prompt(msg: str) -> str:
    print(msg, end="", flush=True)
    return sys.stdin.readline().strip()


def summarize_meta(meta: Dict) -> str:
    intent = meta.get("intent")
    plan = meta.get("plan_type")
    req = meta.get("requirements") or {}
    outs = req.get("outputs") or []
    params = (req.get("params") or {})
    attaches = meta.get("attachments") or []
    lines = [
        f"- intent: {intent}",
        f"- plan_type: {plan}",
        f"- outputs: {outs}",
        f"- params: {params}",
        f"- attachments: {len(attaches)} file(s)",
    ]
    return "\n".join(lines)


def choose_intent() -> str:
    # Deprecated in file-only UX; kept for compatibility
    return ""


def apply_clarifications(meta: Dict) -> Dict:
    # File-only UX: no in-CLI clarification; return meta as-is
    return meta


def spawn_server(script_path: Path):
    python = sys.executable
    cmd = [python, str(script_path), "--mode", "stdio"]
    return {"command": cmd[0], "args": cmd[1:], "env": os.environ.copy()}


def copy_and_snapshot(project_dir: Path, result: Dict, req_path: Path, text: str) -> Dict:
    out_dir = project_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = project_dir / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    artifacts = []
    for p in (result.get("artifacts", []) or []):
        src = Path(p)
        if src.exists():
            dst = out_dir / src.name
            try:
                import shutil

                shutil.copy2(src, dst)
                artifacts.append(str(dst))
            except Exception:
                pass

    import hashlib, platform

    status_seed = (text + (result.get("kind") or "") + "|" + "|".join(result.get("artifacts", []))).encode(
        "utf-8"
    )
    status_hash = hashlib.sha256(status_seed).hexdigest()

    roles = {
        "broker": {"session_dir": os.environ.get("CODEX_SESSION_BASEDIR", "")},
        "boss": {"session_dir": result.get("boss_session_dir")},
        "worker1": {"session_dir": result.get("w1_session_dir")},
        "worker2": {"session_dir": result.get("w2_session_dir")},
    }

    snapshot = {
        "project_id": project_dir.name,
        "requirements_file": str(req_path.resolve()),
        "summary": {"requirement_head": text.strip()[:140], "kind": result.get("kind")},
        "roles": roles,
        "out_dir": str(out_dir.resolve()),
        "status_hash": status_hash,
        "env": {
            "python": sys.version,
            "platform": platform.platform(),
            "CODEX_SESSION_BASEDIR": os.environ.get("CODEX_SESSION_BASEDIR", ""),
        },
    }
    (snapshot_dir / "snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    final_meta = {
        "artifact": artifacts,
        "result": {
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "kind": result.get("kind"),
        },
        "requirements_file": str(req_path.resolve()),
        "project": {"dir": str(project_dir.resolve()), "id": project_dir.name, "status_hash": status_hash},
    }
    out_meta = out_dir / "result.json"
    out_meta.write_text(json.dumps(final_meta, indent=2), encoding="utf-8")
    return {"artifacts": artifacts, "result_json": str(out_meta)}


def _load_codex_config() -> Dict:
    cfg_path = Path.cwd() / ".codex" / "config.json"
    try:
        if cfg_path.exists():
            return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _load_api_provider_key() -> tuple[str | None, str | None]:
    """Force Qwen Plus as the only provider.

    Look for keystone/qwen-plus.key first; fallback to env QWEN_API_KEY.
    Return ("qwen", key) if found; otherwise (None, None).
    """
    ks = Path.cwd() / "keystone"
    try:
        qp = ks / "qwen-plus.key"
        if qp.exists():
            content = qp.read_text(encoding="utf-8").strip()
            key = (content.splitlines()[0] if content else None)
            if key:
                return "qwen", key
    except Exception:
        pass
    key_env = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if key_env:
        return "qwen", key_env
    return None, None


def _load_global_guideline() -> str:
    try:
        g = Path(__file__).parents[3] / "guideline" / "global_rules.md"
        if g.exists():
            return g.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def llm_analyze_requirement(text: str, attachments: List[Dict] | None = None) -> Dict:
    import re
    from camel.agents import ChatAgent
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    provider_hint, api_key = _load_api_provider_key()
    if not api_key:
        raise RuntimeError("缺少 Qwen-Plus API Key：请在 keystone/qwen-plus.key 或环境变量 QWEN_API_KEY 配置")
    # Force Qwen Plus
    model = ModelFactory.create(model_platform=ModelPlatformType.QWEN, model_type="qwen-plus", api_key=api_key)
    guideline = _load_global_guideline()
    sys_prompt = (
        ("[GLOBAL_GUIDELINE]\n" + guideline + "\n\n" if guideline else "") +
        "You are a strict planning assistant. Convert the user's free-form requirement into a single JSON object (no prose).\n"
        "Output strictly one JSON object with keys: intent, plan_type, requirements.\n"
        "For plan_type=python: requirements MUST include {script_name, script_content, outputs[], pip?}. Do not rely on templates.\n"
        "For plan_type=bash: requirements MUST include {script_name, script_content, outputs[]}.\n"
        "For plan_type=c: requirements MUST include {filename, content, binary, expected_output?}.\n"
        "No explanations outside JSON."
    )
    agent = ChatAgent(system_message=sys_prompt, model=model)
    att_section = ""
    if attachments:
        att_lines = [f"- {a.get('name')} => {a.get('path')}" for a in attachments]
        att_section = "\nAttached files:\n" + "\n".join(att_lines)
    user_msg = f"Requirement:\n{text}\n{att_section}\nOutput JSON only."
    resp = agent.step(user_msg)
    if resp.terminated or not resp.msgs:
        raise RuntimeError("GPT 解析失败：未返回内容")
    content = resp.msgs[0].content.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", content)
    json_text = m.group(0) if m else content
    meta = json.loads(json_text)
    if not isinstance(meta, dict):
        raise RuntimeError("GPT 解析失败：返回非 JSON 对象")
    meta.setdefault("raw_text", text)
    return meta


# (removed) llm_analyze_requirement: we use Codex CLI heuristic analyzer only


def main():
    parser = argparse.ArgumentParser(description="Interactive Broker CLI (Codex CLI Session)")
    parser.add_argument("--requirement", "-r", help="Path to requirement.md (optional)")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    DBG = bool(args.debug or os.environ.get("CODEX_DEBUG"))

    # Ensure private role workdir with strict permissions
    role_priv = Path.cwd() / "workdir" / "broker"
    role_priv.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(role_priv, 0o700)
    except Exception:
        pass

    # Ensure public attachments directory and export ATTACHMENTS_DIR
    public_dir = Path.cwd() / "attachments"
    public_dir.mkdir(parents=True, exist_ok=True)
    try:
        # public rw for all agents
        os.chmod(public_dir, 0o777)
    except Exception:
        pass
    os.environ["ATTACHMENTS_DIR"] = str(public_dir.resolve())

    # Create a dedicated Codex CLI session for the interactive broker
    session = CodexCliSession.create()
    print(json.dumps({"session_dir": str(session.work_dir)}))
    if DBG:
        msg = f"session: {session.work_dir}"
        print(f"[DEBUG][broker] {msg}", flush=True)
        log_progress("broker", msg)

    if args.requirement:
        req_path = Path(args.requirement)
    else:
        rp = prompt("请输入 requirement.md 路径（默认 ./requirement.md）: ") or "requirement.md"
        req_path = Path(rp)

    if not req_path.exists():
        print(f"找不到需求文件: {req_path}")
        sys.exit(1)

    text = req_path.read_text(encoding="utf-8")
    if DBG:
        print("[DEBUG][broker] loaded requirement from:", req_path, flush=True)

    # Load .codex env (if any)
    try:
        env_cfg = _load_codex_config().get("env", {})
        for k, v in (env_cfg or {}).items():
            os.environ.setdefault(str(k), str(v))
    except Exception:
        pass

    # Analyze requirement strictly via GPT (no rule fallback)
    attachments: List[Dict] = []
    try:
        pub = Path(os.environ.get("ATTACHMENTS_DIR") or (Path.cwd()/"attachments")).resolve()
        if pub.exists():
            for p in pub.glob("**/*"):
                if p.is_file() and not p.name.startswith('.'):
                    attachments.append({"name": p.name, "path": str(p)})
    except Exception:
        pass
    try:
        meta = llm_analyze_requirement(text, attachments)
    except Exception as e:
        print("[Broker] 解析 requirement.md 失败（LLM）：", str(e))
        print("请配置 Qwen-Plus API Key 或修正 requirement.md 后重试。")
        sys.exit(2)
    if (not isinstance(meta, dict)) or (not meta.get("intent")) or (not meta.get("plan_type")) or (not meta.get("requirements")):
        print("[Broker] 解析结果不完整：缺少 intent/plan_type/requirements 任一项，请修改 requirement.md 后重试。")
        sys.exit(2)

    print("[Broker] 分析 requirement.md：没问题，开始工作……")
    if DBG:
        summary = summarize_meta(meta)
        print("[DEBUG][broker] meta summary:\n" + summary, flush=True)
        log_progress("broker", summary)

    # Prepare project directory by title
    title = extract_title(text)
    slug = slugify(title)
    import hashlib

    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    project_root = Path.cwd() / "project"
    project_root.mkdir(exist_ok=True)
    project_dir = project_root / f"{slug}-{h}"
    (project_dir / "out").mkdir(parents=True, exist_ok=True)
    (project_dir / "snapshot").mkdir(parents=True, exist_ok=True)
    if DBG:
        log_dir = project_dir / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CODEX_LOG_DIR"] = str(log_dir.resolve())
        log_progress("broker", f"project dir: {project_dir}")
    # ensure per-project requirement.txt for venv management
    try:
        reqm = (meta.get("requirements") or {})
        pip_deps = reqm.get("pip") or []
        req_txt = project_dir / "requirement.txt"
        if not req_txt.exists():
            req_txt.write_text("\n".join(str(p) for p in pip_deps), encoding="utf-8")
        if DBG:
            print("[DEBUG][broker] project:", project_dir, "requirements deps:", pip_deps, flush=True)
    except Exception:
        pass

    # Set per-project session base for all downstream agents
    os.environ["CODEX_SESSION_BASEDIR"] = str((project_dir / ".camel_sessions").resolve())

    # Persist MetaInfo for boss to read (and for audit)
    meta_path = project_dir / "snapshot" / "meta.json"
    try:
        (project_dir / "snapshot").mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Broker] 已生成 MetaInfo: {meta_path}")
    except Exception as e:
        print("[Broker] 写入 MetaInfo 失败：", str(e))
        sys.exit(2)

    # Spawn boss and execute meta
    boss_script = Path(__file__).parents[1] / "codex_cli_boss_mcp" / "boss_server.py"
    boss_cfg = spawn_server(boss_script)

    async def run_boss_once(m: Dict) -> Dict:
        async with MCPClient(boss_cfg, timeout=120.0) as boss:
            res_raw = await boss.call_tool("execute", {"meta": m})
            content = getattr(res_raw, "content", [None])[0]
            return json.loads(getattr(content, "text", "{}"))

    import asyncio

    result = asyncio.run(run_boss_once(meta))
    if DBG:
        msg = f"boss result keys: {list(result.keys())}"
        print(f"[DEBUG][broker] {msg}", flush=True)
        log_progress("broker", msg)
    # File-only UX: if environment required, instruct user to update requirement.md and exit
    if not result.get("ok") and result.get("action_required") == "environment":
        print("[Boss→Broker] 需要准备环境，已暂停执行。")
        print("建议在 requirement.md 的 Environment 或 Params 段补充以下内容：")
        if result.get("message"):
            print("- 提示：", result.get("message"))
        for need in (result.get("needs") or []):
            print("- 需求：", need)
        print("请修改 requirement.md 后重新运行：python3 run.py requirement.md")
        sys.exit(3)

    # Copy artifacts and snapshot
    copied = copy_and_snapshot(project_dir, result, req_path, text)
    if result.get("ok"):
        print("执行完成。产物：")
        for p in copied["artifacts"]:
            print("- ", p)
    else:
        print("执行未完成，详情见：", copied["result_json"])
    if DBG:
        log_progress("broker", "execution finished")


if __name__ == "__main__":
    main()
