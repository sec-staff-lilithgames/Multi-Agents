import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence


@dataclass
class CodexCliSession:
    """Represents an isolated session environment.

    Each session has its own working directory and Python virtualenv to
    provide best-effort isolation for tools and file operations.

    Note: This provides process and filesystem separation within a user
    workspace. For hard isolation (namespaces/containers), integrate with a
    stronger sandbox at the launcher layer.
    """

    session_id: str
    root_dir: Path
    work_dir: Path
    venv_dir: Path

    @classmethod
    def create(cls, base_dir: Optional[Path] = None) -> "CodexCliSession":
        # Allow overriding the session base via env for project isolation
        env_base = os.environ.get("CODEX_SESSION_BASEDIR")
        if env_base and not base_dir:
            base = Path(env_base)
        else:
            base = base_dir or Path.cwd() / ".camel_sessions"
        base.mkdir(parents=True, exist_ok=True)
        sid = str(uuid.uuid4())
        work_dir = base / f"session-{sid}"
        venv_dir = work_dir / ".venv"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Create dedicated venv
        py = sys.executable
        subprocess.run([py, "-m", "venv", str(venv_dir)], check=True)

        return cls(session_id=sid, root_dir=base, work_dir=work_dir, venv_dir=venv_dir)

    def python_bin(self) -> Path:
        if sys.platform == "win32":
            return self.venv_dir / "Scripts" / "python.exe"
        return self.venv_dir / "bin" / "python"

    def pip_bin(self) -> Path:
        if sys.platform == "win32":
            return self.venv_dir / "Scripts" / "pip.exe"
        return self.venv_dir / "bin" / "pip"

    def run(self,
            cmd: Sequence[str],
            env: Optional[Dict[str, str]] = None,
            cwd: Optional[Path] = None,
            check: bool = True,
            timeout: Optional[float] = None) -> subprocess.CompletedProcess:
        """Run a command inside the session's environment.

        - Activates the session venv by preferring its bin in PATH
        - Uses the session work_dir as default cwd
        """
        full_env = os.environ.copy()
        # Prepend venv bin to PATH for isolation
        bin_dir = self.python_bin().parent
        full_env["PATH"] = f"{bin_dir}:{full_env.get('PATH','')}"
        if env:
            full_env.update(env)
        return subprocess.run(
            list(cmd),
            cwd=str(cwd or self.work_dir),
            env=full_env,
            text=True,
            capture_output=True,
            check=check,
            timeout=timeout,
        )

    def cleanup(self) -> None:
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)
