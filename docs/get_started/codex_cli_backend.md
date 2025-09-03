# Codex CLI Backend (Session-Isolated)

This backend introduces a new platform `codex-cli` that binds each model instance to a fully isolated session. Each session has its own working directory and Python virtual environment, offering process-level and filesystem-level separation for tools, interpreters, and code execution.

What this change does
- New platform: `ModelPlatformType.CODEXCLI` (value: `"codex-cli"`).
- New model: `CodexCliModel` with a per-instance `CodexCliSession`.
- Isolation: Each session creates a unique folder under `.camel_sessions/session-<uuid>` and a fresh `.venv` inside it. Commands and tools can run within this sandbox via the session.
- Compatibility: The model returns a placeholder `ChatCompletion` so it can be used anywhere a `BaseModelBackend` is expected. LLM transport is intentionally decoupled and can be integrated on top of the session.

Quickstart
```bash
source .venv/bin/activate
python examples/usecases/codex_cli_quickstart.py
```

You should see a message indicating the session’s isolated working directory and `session_id`.

Programmatic usage
```python
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

model = ModelFactory.create(
    model_platform=ModelPlatformType.CODEXCLI,
    model_type=ModelType.STUB,  # token counting compatible, no external API
)

resp = model.run([{"role": "user", "content": "hello"}])
print(resp.choices[0].message.content)
```

Running tools in the session
If you want to execute commands/files in the isolated environment, access the session from the model:
```python
codex_model = model  # created as above
session = codex_model.session

# Install a package only inside this session
session.run([str(session.pip_bin()), "install", "requests"]).check_returncode()

# Run a Python snippet inside the session venv and work dir
python = str(session.python_bin())
session.run([python, "-c", "import sys; print(sys.executable)"])
```

Notes and next steps
- The backend intentionally does not make network calls to LLM services; it focuses on isolation primitives. To integrate Codex CLI message transport, wire your dispatcher to use the session’s working directory and environment.
- For stronger isolation (namespaces/containers), adapt `CodexCliSession.create()` to launch within your preferred sandbox.

