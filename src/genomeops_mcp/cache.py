"""Simple file-backed cache with TTL for HTTP responses and schemas."""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "genomeops-mcp"


def _cache_path(key: str) -> Path:
    safe = key.replace("/", "_").replace(":", "_")
    return _CACHE_DIR / f"{safe}.json"


def cache_get(key: str, ttl_seconds: int) -> dict | list | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        if time.time() - raw["_ts"] > ttl_seconds:
            logger.debug("Cache expired for %s", key)
            return None
        logger.debug("Cache hit for %s", key)
        return raw["data"]
    except Exception:
        return None


def cache_set(key: str, data: dict | list) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(key)
    try:
        path.write_text(json.dumps({"_ts": time.time(), "data": data}))
    except Exception as exc:
        logger.warning("Failed to write cache for %s: %s", key, exc)


def cache_clear(key: str | None = None) -> None:
    """Clear a specific cache entry, or the entire cache if key is None."""
    if key is not None:
        _cache_path(key).unlink(missing_ok=True)
    else:
        for f in _CACHE_DIR.glob("*.json"):
            f.unlink(missing_ok=True)
