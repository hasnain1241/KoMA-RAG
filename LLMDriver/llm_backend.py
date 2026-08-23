"""
Pluggable LLM backend for KoMA / KoMA-RAG.

Supports OPENAI_API_TYPE values:
  - azure, openai, ollama  (existing langchain paths)
  - nvidia / nvidia_nim    (OpenAI SDK → NVIDIA Integrate API)

No silent Mock fallback: real providers raise on API errors.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Union


MessageLike = Union[Any, Dict[str, str]]


class LLMResponse:
    """Minimal response object compatible with langchain Message.content usage."""

    def __init__(self, content: str):
        self.content = content


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(key: str, default: Optional[float] = None) -> Optional[float]:
    raw = _env(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _role_model(role: str) -> str:
    """Resolve model id for a logical role (driver/master/reflection/verify)."""
    role = (role or "driver").lower()
    mapping = {
        "driver": _env("MODEL_DRIVER") or _env("CHATGPT_MODEL") or "meta/llama-3.1-70b-instruct",
        "planner": _env("MODEL_DRIVER") or _env("CHATGPT_MODEL") or "meta/llama-3.1-70b-instruct",
        "master": _env("MODEL_MASTER") or _env("CHATGPT_MODEL") or "mistralai/mistral-medium-3.5-128b",
        "reflection": _env("MODEL_REFLECTION") or _env("MODEL_VERIFY") or _env("CHATGPT_MODEL")
                      or "mistralai/mistral-medium-3.5-128b",
        "verify": _env("MODEL_VERIFY") or _env("MODEL_REFLECTION") or _env("CHATGPT_MODEL")
                  or "mistralai/mistral-medium-3.5-128b",
    }
    return mapping.get(role, mapping["driver"])


def _is_deepseek(model: str) -> bool:
    return "deepseek" in (model or "").lower()


def _message_role_content(msg: MessageLike) -> Dict[str, str]:
    if isinstance(msg, dict):
        return {"role": msg["role"], "content": msg["content"]}
    # langchain Message duck-typing
    cls = msg.__class__.__name__
    if cls == "SystemMessage":
        role = "system"
    elif cls == "AIMessage":
        role = "assistant"
    else:
        role = "user"
    return {"role": role, "content": msg.content}


def _to_openai_messages(messages: Sequence[MessageLike]) -> List[Dict[str, str]]:
    return [_message_role_content(m) for m in messages]


class NvidiaChatLLM:
    """
    Generic OpenAI-v1-SDK-compatible chat client.

    Despite the name (kept for backward compat), this works against any
    OpenAI-compatible endpoint: NVIDIA Integrate API, Groq, real OpenAI, etc.
    Matches:
      client = OpenAI(base_url="...", api_key="...")
      client.chat.completions.create(..., stream=False)
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        top_p: Optional[float] = None,
        thinking: Optional[bool] = None,
        provider_label: str = "NVIDIA",
    ) -> None:
        self.model = model
        self.provider_label = provider_label
        # Prefer explicit api_key argument; treat blank as missing
        if api_key is not None and str(api_key).strip() != "":
            self.api_key = str(api_key).strip()
        else:
            self.api_key = _env("NVIDIA_API_KEY") or _env("OPENAI_API_KEY")
        self.base_url = (base_url or _env("NVIDIA_BASE_URL") or _env("OPENAI_API_BASE")
                         or "https://integrate.api.nvidia.com/v1").rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # top_p: config/env, else 0.95 for DeepSeek (NVIDIA sample default), else omit
        if top_p is not None:
            self.top_p = top_p
        else:
            env_top_p = _env_float("NVIDIA_TOP_P")
            if env_top_p is not None:
                self.top_p = env_top_p
            elif _is_deepseek(model):
                self.top_p = 0.95
            else:
                self.top_p = None

        # DeepSeek: always send chat_template_kwargs; default thinking=False (AV hot path)
        if thinking is None:
            thinking = _env_bool("NVIDIA_THINKING", False)
        self.thinking = bool(thinking)

        if (
            not self.api_key
            or str(self.api_key).strip().startswith("xxxx")
            or str(self.api_key).strip().startswith("CHANGEME")
            or str(self.api_key).strip() in ("nvapi-...", "nvapi-")
        ):
            raise RuntimeError(
                f"{self.provider_label} API key missing. Set NVIDIA_API_KEY / GROQ_API_KEY "
                "(or OPENAI_API_KEY) in the environment, or in config.yaml."
            )

        from openai import OpenAI

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def __call__(self, messages: Sequence[MessageLike], **kwargs: Any) -> LLMResponse:
        return self.invoke(messages, **kwargs)

    def invoke(self, messages: Sequence[MessageLike], **kwargs: Any) -> LLMResponse:
        create_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False,
        }
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            create_kwargs["top_p"] = top_p

        # DeepSeek (and any deepseek-* id): always pass thinking via extra_body
        if _is_deepseek(self.model):
            thinking = kwargs.get("thinking", self.thinking)
            create_kwargs["extra_body"] = {
                "chat_template_kwargs": {"thinking": bool(thinking)}
            }

        max_retries = _env_int("LLM_MAX_RETRIES", 5)
        base_delay = _env_float("LLM_RETRY_BASE_DELAY", 5.0) or 5.0
        max_delay = _env_float("LLM_RETRY_MAX_DELAY", 60.0) or 60.0
        attempt = 0
        while True:
            try:
                completion = self._client.chat.completions.create(**create_kwargs)
                break
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                is_rate_limit = status_code == 429 or exc.__class__.__name__ == "RateLimitError"
                if not is_rate_limit or attempt >= max_retries:
                    raise RuntimeError(
                        f"{self.provider_label} API request failed for model={self.model}: {exc}"
                    ) from exc

                retry_after = None
                response = getattr(exc, "response", None)
                headers = getattr(response, "headers", None)
                if headers is not None:
                    retry_after = headers.get("retry-after")
                try:
                    wait_s = float(retry_after) if retry_after else base_delay * (2 ** attempt)
                except (TypeError, ValueError):
                    wait_s = base_delay * (2 ** attempt)
                wait_s = min(wait_s, max_delay)

                attempt += 1
                print(
                    f"[rate-limit] {self.provider_label} 429 for model={self.model}; "
                    f"retry {attempt}/{max_retries} in {wait_s:.1f}s..."
                )
                time.sleep(wait_s)

        try:
            content = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"{self.provider_label} API returned unexpected payload for model={self.model}: {completion!r}"
            ) from exc

        if content is None:
            raise RuntimeError(
                f"{self.provider_label} API returned empty content for model={self.model}"
            )
        return LLMResponse(str(content))


class LangchainChatAdapter:
    """Thin adapter so azure/openai/ollama share the same call surface."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def __call__(self, messages: Sequence[MessageLike], **kwargs: Any) -> Any:
        return self.invoke(messages, **kwargs)

    def invoke(self, messages: Sequence[MessageLike], **kwargs: Any) -> Any:
        from langchain.schema import AIMessage, HumanMessage, SystemMessage
        converted = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    converted.append(SystemMessage(content=content))
                elif role == "assistant":
                    converted.append(AIMessage(content=content))
                else:
                    converted.append(HumanMessage(content=content))
            else:
                converted.append(msg)
        return self.llm(converted)


def create_chat_llm(
    role: str = "driver",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    request_timeout: float = 60.0,
) -> Any:
    """
    Factory for role-specific chat models.

    Raises clearly on unknown API type or missing credentials (no Mock fallback).

    max_tokens: callers (driver/master/verify) pass role-specific budgets.
    If omitted under nvidia, uses NVIDIA_MAX_TOKENS (default 2048).
    DeepSeek NVIDIA chat demos often use 16384 — do not use that on the AV hot path.
    """
    api_type = (_env("OPENAI_API_TYPE") or "openai").lower()
    model = _role_model(role)

    if max_tokens is None:
        if api_type in ("nvidia", "nvidia_nim"):
            env_max = _env("NVIDIA_MAX_TOKENS")
            try:
                max_tokens = int(env_max) if env_max and str(env_max).strip() else 2048
            except ValueError:
                max_tokens = 2048
        else:
            max_tokens = 2000

    if api_type in ("nvidia", "nvidia_nim"):
        print(f"Using NVIDIA Integrate API (role={role}, model={model})")
        return NvidiaChatLLM(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=request_timeout,
            provider_label="NVIDIA",
        )

    if api_type == "openai":
        # NOTE: deliberately bypasses langchain.chat_models.ChatOpenAI here.
        # langchain==0.0.331 targets the pre-v1 openai SDK, while this repo
        # pins openai>=1.12,<2 for the NVIDIA path; the two are incompatible.
        # Reuse the same direct v1-client path instead (works for Groq,
        # real OpenAI, or any other OpenAI-compatible base_url).
        base_url = _env("OPENAI_API_BASE") or "https://api.openai.com/v1"
        print(f"Using OpenAI-compatible Chat API (role={role}, model={model}, base_url={base_url})")
        return NvidiaChatLLM(
            model=model,
            api_key=_env("OPENAI_API_KEY"),
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=request_timeout,
            provider_label="OpenAI-compatible",
        )

    from langchain.chat_models import AzureChatOpenAI, ChatOllama

    if api_type == "azure":
        print(f"Using Azure Chat API (role={role})")
        deployment = _env("CHATGPT_MODEL") or "GPT-16"
        return LangchainChatAdapter(
            AzureChatOpenAI(
                deployment_name=deployment,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=int(request_timeout),
            )
        )

    if api_type == "ollama":
        ollama_model = _env("OLLAMA_MODEL") or "llama3"
        print(f"Using Ollama (role={role}, model={ollama_model})")
        return LangchainChatAdapter(ChatOllama(model=ollama_model, temperature=temperature))

    raise ValueError(
        f"Unknown OPENAI_API_TYPE={api_type!r}. "
        "Expected one of: azure, openai, ollama, nvidia, nvidia_nim."
    )


def get_embedding_function():
    """
    Return a langchain-compatible embedding function.

    For nvidia/ollama (no OpenAI embedding key required), use local sentence-transformers.
    """
    api_type = (_env("OPENAI_API_TYPE") or "openai").lower()
    emb_backend = (_env("EMBEDDING_BACKEND") or "").lower()

    use_local = emb_backend in ("local", "sentence-transformers", "st", "auto") or api_type in (
        "nvidia", "nvidia_nim", "ollama"
    )
    # Prefer OpenAI/Azure embeddings only when explicitly requested and API type matches
    if emb_backend in ("openai", "azure"):
        use_local = False

    if use_local:
        from langchain.embeddings import HuggingFaceEmbeddings
        model_name = _env("LOCAL_EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Using local embeddings: {model_name}")
        return HuggingFaceEmbeddings(model_name=model_name)

    from langchain.embeddings.openai import OpenAIEmbeddings
    if api_type == "azure":
        return OpenAIEmbeddings(deployment=_env("EMBEDDING_MODEL"), chunk_size=1)
    if api_type == "openai":
        return OpenAIEmbeddings()
    raise ValueError(
        f"No embedding backend configured for OPENAI_API_TYPE={api_type}. "
        "Set EMBEDDING_BACKEND=local or use azure/openai."
    )
