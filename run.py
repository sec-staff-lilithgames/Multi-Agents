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
#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run interactive Broker (Codex CLI)"
    )
    parser.add_argument(
        "requirement",
        nargs="?",
        default="requirement.md",
        help="Path to requirement.md (default: requirement.md)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode to print agent progress",
    )
    args = parser.parse_args()

    broker_cli = (
        Path(__file__).parent
        / "examples"
        / "usecases"
        / "codex_cli_broker_mcp"
        / "broker_cli.py"
    )
    if not broker_cli.exists():
        print(f"Broker CLI not found: {broker_cli}")
        sys.exit(1)

    # Prefer a modern Python runtime for the child process (>=3.10)
    py = sys.executable
    try:
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

    debug = bool(args.debug or os.environ.get("CODEX_DEBUG"))
    cmd = [py, str(broker_cli), "-r", str(Path(args.requirement).resolve())]
    if debug:
        cmd.append("-d")
    try:
        # Ensure repo root is importable by child script (for `camel` package)
        env = dict(**os.environ)
        repo_root = str(Path(__file__).parent.resolve())
        old_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_root}:{old_pp}" if old_pp else repo_root
        if debug:
            env["CODEX_DEBUG"] = "1"
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
