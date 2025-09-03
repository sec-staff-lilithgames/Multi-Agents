import time
from typing import Any, Dict, List, Optional, Type, Union

from openai import AsyncStream, Stream
from pydantic import BaseModel

from camel.messages import OpenAIMessage
from camel.models import BaseModelBackend
from camel.models.stub_model import StubTokenCounter
from camel.runtime.codex_cli_session import CodexCliSession
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
    ModelType,
)
from camel.utils import BaseTokenCounter


class CodexCliModel(BaseModelBackend):
    """Backend adapter that binds each model instance to an isolated
    Codex CLI-like session environment.

    Notes
    - This backend is transport-agnostic for LLM inference. It does not call
      remote APIs. Instead, it focuses on providing strong per-session process
      and filesystem isolation that other components (tools, interpreters,
      external runners) can rely on.
    - Message generation here is a placeholder to keep CAMEL interfaces stable.
      Integrate your Codex CLI message transport on top of the session if
      needed.
    """

    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            model_type,
            model_config_dict,
            api_key,
            url,
            token_counter,
            timeout,
            max_retries,
        )
        # Create an isolated session per model instance
        self._session = CodexCliSession.create()

    @property
    def session(self) -> CodexCliSession:
        return self._session

    @property
    def token_counter(self) -> BaseTokenCounter:
        if not self._token_counter:
            # Neutral, non-API-dependent counter
            self._token_counter = StubTokenCounter()
        return self._token_counter

    def _mk_placeholder_response(self, content: str) -> ChatCompletion:
        return ChatCompletion(
            id=f"codex-cli-{self._session.session_id}",
            model=str(self.model_type),
            object="chat.completion",
            created=int(time.time()),
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content=content,
                        role="assistant",
                    ),
                    logprobs=None,
                )
            ],
            usage=CompletionUsage(
                completion_tokens=10,
                prompt_tokens=10,
                total_tokens=20,
            ),
        )

    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        # Placeholder response; attach session metadata
        note = (
            "[CodexCLI backend] Session initialized with isolated venv and work"
            f" dir: {self._session.work_dir} (session_id={self._session.session_id})."
        )
        return self._mk_placeholder_response(note)

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        # Placeholder response; attach session metadata
        note = (
            "[CodexCLI backend] Session initialized with isolated venv and work"
            f" dir: {self._session.work_dir} (session_id={self._session.session_id})."
        )
        return self._mk_placeholder_response(note)

