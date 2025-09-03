#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run interactive Broker (Codex CLI)")
    parser.add_argument("requirement", nargs="?", default="requirement.md", help="Path to requirement.md (default: requirement.md)")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode to print agent progress")
    args = parser.parse_args()

    broker_cli = Path(__file__).parent / "examples" / "usecases" / "codex_cli_broker_mcp" / "broker_cli.py"
    if not broker_cli.exists():
        print(f"Broker CLI not found: {broker_cli}")
        sys.exit(1)

    # Prefer a modern Python runtime for the child process (>=3.10)
    py = sys.executable
    try:
        import platform as _platform
        v_info = sys.version_info
        if v_info < (3, 10):
            # Prefer project local venv if present
            venv_py = Path(__file__).parent / ".py312-venv" / "bin" / "python"
            if venv_py.exists():
                py = str(venv_py)
            else:
                # Try common Homebrew paths
                for cand in [
                    "/opt/homebrew/bin/python3.12",
                    "/opt/homebrew/bin/python3.11",
                    "/usr/local/bin/python3.12",
                    "/usr/local/bin/python3.11",
                ]:
                    if os.path.exists(cand):
                        py = cand
                        break
    except Exception:
        pass

    cmd = [py, str(broker_cli), "-r", str(Path(args.requirement).resolve())]
    if args.debug:
        cmd.append("-d")
    try:
        # Ensure repo root is importable by child script (for `camel` package)
        env = dict(**os.environ)
        repo_root = str(Path(__file__).parent.resolve())
        old_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_root}:{old_pp}" if old_pp else repo_root
        if args.debug:
            env["CODEX_DEBUG"] = "1"
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
