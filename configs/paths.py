"""Resolve model / tool paths from env, configs/paths.yaml, or defaults.

Priority: environment variable > configs/paths.yaml > built-in default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PATHS_YAML = PROJECT_ROOT / "configs" / "paths.yaml"
_CACHE: Optional[Dict[str, Any]] = None

# Env var names aligned with paths.example.yaml keys (upper snake).
_ENV_MAP = {
    "qwen2_audio_model": "QWEN2_AUDIO_MODEL",
    "step_audio_root": "STEP_AUDIO_ROOT",
    "step_audio_model": "STEP_AUDIO_MODEL",
    "matcha_root": "MATCHA_ROOT",
    "cosyvoice_root": "COSYVOICE_ROOT",
    "cosyvoice_model": "COSYVOICE_MODEL",
    "project_root": "PROJECT_ROOT",
}

_DEFAULTS = {
    "project_root": str(PROJECT_ROOT),
    "qwen2_audio_model": os.environ.get("QWEN2_AUDIO_MODEL", "/models/Qwen2-Audio-7B-Instruct"),
    "step_audio_root": os.environ.get("STEP_AUDIO_ROOT", "/opt/Step-Audio2"),
    "step_audio_model": os.environ.get(
        "STEP_AUDIO_MODEL", "/models/Step-Audio-2-mini"
    ),
    "matcha_root": os.environ.get("MATCHA_ROOT", "/opt/Matcha-TTS"),
    "cosyvoice_root": os.environ.get("COSYVOICE_ROOT", "/opt/CosyVoice"),
    "cosyvoice_model": os.environ.get("COSYVOICE_MODEL", "/models/CosyVoice2-0.5B"),
    "path_rewrites": [],
}


def _load_yaml() -> Dict[str, Any]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    data: Dict[str, Any] = {}
    if _PATHS_YAML.exists():
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(_PATHS_YAML.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}
    _CACHE = data
    return data


def get_path(key: str, default: Optional[str] = None) -> str:
    env_name = _ENV_MAP.get(key)
    if env_name:
        val = os.environ.get(env_name, "").strip()
        if val:
            return val
    yaml_data = _load_yaml()
    if key in yaml_data and yaml_data[key]:
        return str(yaml_data[key])
    if default is not None:
        return default
    return str(_DEFAULTS.get(key, ""))


def path_rewrites() -> List[Tuple[str, str]]:
    yaml_data = _load_yaml()
    raw = yaml_data.get("path_rewrites") or _DEFAULTS.get("path_rewrites") or []
    out: List[Tuple[str, str]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]), str(item[1])))
    # Always allow optional env-style rewrite pair
    old = os.environ.get("PATH_REWRITE_OLD", "").strip()
    new = os.environ.get("PATH_REWRITE_NEW", "").strip()
    if old and new:
        out.append((old, new))
    return out


def resolve_existing_path(path: str) -> str:
    """Apply configured prefix rewrites until an existing path is found (or return last)."""
    candidates = [path]
    for old, new in path_rewrites():
        if path.startswith(old):
            candidates.append(new + path[len(old) :])
    for cand in candidates:
        if Path(cand).exists():
            return cand
    return candidates[-1] if candidates else path
