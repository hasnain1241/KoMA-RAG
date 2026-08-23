
import os
from typing import Any, Dict

import yaml


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader) or {}
    return cfg


def apply_config_to_env(cfg: Dict[str, Any]) -> None:
    """Push config keys into process env (env vars already set win)."""
    def set_if_absent(key: str, value: Any) -> None:
        if value is None:
            return
        if key not in os.environ or os.environ.get(key, "") == "":
            os.environ[key] = str(value)

    api_type = str(cfg.get("OPENAI_API_TYPE", "openai")).lower()
    os.environ["OPENAI_API_TYPE"] = api_type

    if api_type == "azure":
        set_if_absent("OPENAI_API_VERSION", cfg.get("OPENAI_API_VERSION"))
        set_if_absent("OPENAI_API_BASE", cfg.get("OPENAI_API_BASE"))
        set_if_absent("OPENAI_API_KEY", cfg.get("OPENAI_API_KEY"))
        set_if_absent("EMBEDDING_MODEL", cfg.get("EMBEDDING_MODEL"))
        set_if_absent("CHATGPT_MODEL", cfg.get("CHATGPT_MODEL"))
    elif api_type == "openai":
        # Groq (or any OpenAI-compatible provider) key: env var wins over config.yaml.
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        if groq_key:
            os.environ["OPENAI_API_KEY"] = groq_key
        else:
            set_if_absent("OPENAI_API_KEY", cfg.get("REAL_OPENAI_KEY") or cfg.get("OPENAI_API_KEY"))
        set_if_absent("OPENAI_API_BASE", cfg.get("OPENAI_API_BASE"))
        set_if_absent("CHATGPT_MODEL", cfg.get("CHATGPT_MODEL"))
    elif api_type in ("nvidia", "nvidia_nim"):
        # Prefer NVIDIA_API_KEY from env, else config (ignore placeholders)
        def _clean_key(val: Any) -> str:
            if val is None:
                return ""
            s = str(val).strip()
            if not s or s.startswith("xxxx") or s in ("nvapi-...", "nvapi-"):
                return ""
            return s

        nvidia_key = _clean_key(os.environ.get("NVIDIA_API_KEY")) or _clean_key(cfg.get("NVIDIA_API_KEY"))
        if not nvidia_key:
            nvidia_key = _clean_key(cfg.get("OPENAI_API_KEY")) or _clean_key(cfg.get("REAL_OPENAI_KEY"))
        os.environ["NVIDIA_API_KEY"] = nvidia_key
        os.environ["OPENAI_API_KEY"] = nvidia_key
        base = cfg.get("NVIDIA_BASE_URL") or cfg.get("OPENAI_API_BASE") or \
            "https://integrate.api.nvidia.com/v1"
        os.environ["NVIDIA_BASE_URL"] = str(base)
        os.environ["OPENAI_API_BASE"] = str(base)
        set_if_absent("CHATGPT_MODEL", cfg.get("CHATGPT_MODEL"))
        set_if_absent("MODEL_DRIVER", cfg.get("MODEL_DRIVER"))
        set_if_absent("MODEL_MASTER", cfg.get("MODEL_MASTER"))
        set_if_absent("MODEL_REFLECTION", cfg.get("MODEL_REFLECTION"))
        set_if_absent("MODEL_VERIFY", cfg.get("MODEL_VERIFY"))
        set_if_absent("NVIDIA_TOP_P", cfg.get("NVIDIA_TOP_P"))
        set_if_absent("NVIDIA_THINKING", cfg.get("NVIDIA_THINKING"))
        set_if_absent("NVIDIA_MAX_TOKENS", cfg.get("NVIDIA_MAX_TOKENS"))
    elif api_type == "ollama":
        set_if_absent("OLLAMA_MODEL", cfg.get("OLLAMA_MODEL", "llama3"))
    else:
        raise ValueError(f"Unknown OPENAI_API_TYPE: {api_type}")

    # Shared model / embedding settings
    for key in (
        "CHATGPT_MODEL", "MODEL_DRIVER", "MODEL_MASTER", "MODEL_REFLECTION",
        "MODEL_VERIFY", "EMBEDDING_BACKEND", "LOCAL_EMBEDDING_MODEL",
        "EMBEDDING_MODEL", "OLLAMA_MODEL",
    ):
        if key in cfg:
            set_if_absent(key, cfg.get(key))


def load_openai_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Backward-compatible entry: load yaml and configure environment."""
    cfg = load_config(path)
    apply_config_to_env(cfg)
    return cfg


def get_framework_flags(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ablation / pipeline knobs matching the manuscript."""
    return {
        "USE_MEMORY": _as_bool(cfg.get("USE_MEMORY"), False),
        "REFLECTION": _as_bool(cfg.get("REFLECTION"), False),
        "ENABLE_MASTER": _as_bool(cfg.get("ENABLE_MASTER"), False),
        "ENABLE_VERIFICATION": _as_bool(cfg.get("ENABLE_VERIFICATION"), False),
        "MASTER_USE_LLM": _as_bool(cfg.get("MASTER_USE_LLM"), True),
        "MASTER_CONFLICT_ONLY_LLM": _as_bool(cfg.get("MASTER_CONFLICT_ONLY_LLM"), True),
        "FACTUAL_USE_LLM": _as_bool(cfg.get("FACTUAL_USE_LLM"), False),
        "TAU_VERIFY": _as_float(cfg.get("TAU_VERIFY"), 0.5),
        "LAMBDA_TIME": _as_float(cfg.get("LAMBDA_TIME"), 0.01),
        "DELTA_T_SAFE": _as_float(cfg.get("DELTA_T_SAFE"), 2.0),
        "LAMBDA_COOP": _as_float(cfg.get("LAMBDA_COOP"), 0.3),
        "LAMBDA_CONFLICT": _as_float(cfg.get("LAMBDA_CONFLICT"), 0.5),
        "VERIFICATION_EMPTY_FALLBACK": str(cfg.get("VERIFICATION_EMPTY_FALLBACK", "similarity")),
        "SCENARIO_TYPE": str(cfg.get("SCENARIO_TYPE", "highway")),
        "FEW_SHOT_NUM": _as_int(cfg.get("FEW_SHOT_NUM"), 2),
        "K_CANDIDATES": _as_int(cfg.get("K_CANDIDATES"), 5),
        "ENABLE_DUAL_PATH": _as_bool(cfg.get("ENABLE_DUAL_PATH"), False),
        "T_MAX_SECONDS": _as_float(cfg.get("T_MAX_SECONDS"), 0.0),
        "ASYNC_REFLECTION": _as_bool(cfg.get("ASYNC_REFLECTION"), False),
    }


if __name__ == "__main__":
    c = load_openai_config()
    print(get_framework_flags(c))
