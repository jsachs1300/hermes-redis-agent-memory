"""Redis Agent Memory Server provider for Hermes Agent.

This module implements the Hermes MemoryProvider interface against Redis Agent
Memory Server (AMS).  The initial provider intentionally keeps the integration
small: long-term-memory recall, working-memory turn sync, explicit remember /
search / forget tools, and mirroring of built-in Hermes memory writes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.memory_provider import MemoryProvider

try:
    from tools.registry import tool_error
except Exception:  # pragma: no cover - used outside Hermes test envs
    def tool_error(message: str) -> str:
        return json.dumps({"error": message})

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_NAMESPACE = "hermes"
_DEFAULT_USER_ID = "hermes-user"
_DEFAULT_SEARCH_MODE = "hybrid"
_VALID_SEARCH_MODES = {"semantic", "keyword", "hybrid"}
_CONFIG_FILENAME = "redis-agent-memory.json"


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _as_int(value: Any, default: int, *, minimum: int = 1, maximum: int = 50) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except Exception:
        return default


def _sanitize_namespace(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.:-]", "-", value or "")
    value = re.sub(r"-+", "-", value).strip("-_.:")
    return value or _DEFAULT_NAMESPACE


def _sanitize_session_id(value: str) -> str:
    """Return a Redis AMS-compatible session id.

    Redis Agent Memory session events only accept alphanumeric characters and
    hyphens. Hermes session ids commonly contain underscores, so normalize the
    session id at the provider boundary before calling add_session_event().
    """
    value = re.sub(r"[^a-zA-Z0-9-]", "-", str(value or ""))
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "hermes-session"


def _sanitize_actor_id(value: str) -> str:
    """Return a Redis AMS-compatible actor id.

    Redis Agent Memory validates ActorID with the same restricted alphabet as
    session ids: alphanumeric characters and hyphens only. Hermes user ids can
    contain platform delimiters such as colons (for example ``system:handoff``),
    so normalize them before calling add_session_event().
    """
    value = re.sub(r"[^a-zA-Z0-9-]", "-", str(value or ""))
    value = re.sub(r"-+", "-", value).strip("-")
    return value or _DEFAULT_USER_ID


def _message_summary(messages: list[dict]) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role") or "unknown")
        content = str(message.get("content") or "")
        parts.append(f"{role}:{len(content)} chars")
    return "[" + ", ".join(parts) + "]"


def _sync_payload_log_details(payload: dict) -> str:
    return (
        f"session_id={payload.get('session_id')} "
        f"original_session_id={payload.get('original_session_id')} "
        f"user_id={payload.get('user_id')} "
        f"namespace={payload.get('namespace')} "
        f"messages={_message_summary(payload.get('messages') or [])}"
    )


def _json_log(value: Any) -> str:
    """Render structured diagnostics for logs without failing on SDK objects."""
    def _default(obj: Any) -> str:
        if hasattr(obj, "value"):
            return str(obj.value)
        return str(obj)

    return json.dumps(value, sort_keys=True, default=_default)


def _content_log_details(content: str) -> dict:
    """Describe content payloads without writing raw conversation text to logs."""
    # Full message text can contain secrets or customer data. Keep the Redis SDK
    # call shape complete while logging length-only content diagnostics.
    return {"type": "Text", "text_chars": len(content)}


def _load_config(hermes_home: str | None = None) -> dict:
    config = {
        "base_url": os.environ.get("REDIS_AGENT_MEMORY_URL", ""),
        "auth_token": os.environ.get("REDIS_AGENT_MEMORY_TOKEN") or os.environ.get("AGENT_MEMORY_API_KEY", ""),
        "user_id": os.environ.get("REDIS_AGENT_MEMORY_USER_ID", _DEFAULT_USER_ID),
        "namespace": os.environ.get("REDIS_AGENT_MEMORY_NAMESPACE", _DEFAULT_NAMESPACE),
        "store_id": os.environ.get("REDIS_AGENT_MEMORY_STORE_ID") or os.environ.get("AGENT_MEMORY_STORE_ID", ""),
        "service_name": os.environ.get("REDIS_AGENT_MEMORY_SERVICE_NAME", ""),
        "search_mode": os.environ.get("REDIS_AGENT_MEMORY_SEARCH_MODE", _DEFAULT_SEARCH_MODE),
        "max_recall_results": os.environ.get("REDIS_AGENT_MEMORY_MAX_RECALL_RESULTS", 8),
        "auto_recall": os.environ.get("REDIS_AGENT_MEMORY_AUTO_RECALL", "true"),
        "auto_sync_turns": os.environ.get("REDIS_AGENT_MEMORY_AUTO_SYNC_TURNS", "true"),
        "api_timeout": os.environ.get("REDIS_AGENT_MEMORY_TIMEOUT", 5.0),
    }

    if hermes_home:
        path = Path(hermes_home) / _CONFIG_FILENAME
        if path.exists():
            try:
                file_config = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(file_config, dict):
                    # File config overrides non-secret env defaults. Secrets stay in env.
                    file_config.pop("auth_token", None)
                    config.update({k: v for k, v in file_config.items() if v is not None and v != ""})
            except Exception:
                logger.debug("Failed to read Redis Agent Memory config", exc_info=True)

    config["base_url"] = str(config.get("base_url") or "").rstrip("/")
    config["auth_token"] = str(config.get("auth_token") or "")
    config["store_id"] = str(config.get("store_id") or "")
    config["service_name"] = str(config.get("service_name") or "")
    config["user_id"] = str(config.get("user_id") or _DEFAULT_USER_ID)
    config["namespace"] = str(config.get("namespace") or _DEFAULT_NAMESPACE)
    mode = str(config.get("search_mode") or _DEFAULT_SEARCH_MODE).lower()
    config["search_mode"] = mode if mode in _VALID_SEARCH_MODES else _DEFAULT_SEARCH_MODE
    config["max_recall_results"] = _as_int(config.get("max_recall_results"), 8, minimum=1, maximum=25)
    config["auto_recall"] = _as_bool(config.get("auto_recall"), True)
    config["auto_sync_turns"] = _as_bool(config.get("auto_sync_turns"), True)
    try:
        config["api_timeout"] = max(0.5, min(30.0, float(config.get("api_timeout", 5.0))))
    except Exception:
        config["api_timeout"] = 5.0
    return config


class _RedisAMSClient:
    """Synchronous Hermes adapter over the official redis-agent-memory SDK.

    The cloud SDK is exposed by the `redis-agent-memory` PyPI package, with
    AgentMemory(server_url, store_id=..., api_key=...). The API key is HTTP
    bearer auth and store_id is the SDK global parameter for all operations.
    """

    def __init__(self, config: dict):
        try:
            from redis_agent_memory import AgentMemory, models
        except ImportError as exc:  # pragma: no cover - exercised in real setup
            raise RuntimeError("redis-agent-memory is not installed. Run: uv pip install redis-agent-memory") from exc

        self._models = models
        self._client = AgentMemory(
            config["base_url"],
            store_id=config.get("store_id") or None,
            api_key=config.get("auth_token") or None,
            timeout_ms=int(float(config.get("api_timeout", 5.0)) * 1000),
        )

    @staticmethod
    def _to_plain(value):
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json", by_alias=False)
        if hasattr(value, "dict"):
            return value.dict()
        return value

    def health(self):
        return self._to_plain(self._client.health())

    def search_long_term_memory(self, **kwargs):
        models = self._models
        filters = models.LongTermMemoryFilter(
            ownerId=models.OwnerIDFilter(eq=kwargs.get("user_id")) if kwargs.get("user_id") else None,
            namespace=models.NamespaceFilter(eq=kwargs.get("namespace")) if kwargs.get("namespace") else None,
        )
        request = models.SearchLongTermMemoryRequestContent(
            text=kwargs.get("text"),
            limit=kwargs.get("limit", 10),
            filter=filters,
        )
        return self._to_plain(self._client.search_long_term_memory(request=request))

    def create_long_term_memory(self, memories):
        models = self._models
        records = []
        for memory in memories:
            memory_type = memory.get("memory_type") or "semantic"
            records.append(models.CreateMemoryRecord(
                id=memory.get("id") or str(uuid.uuid4()),
                text=memory["text"],
                memoryType=models.MemoryType(memory_type),
                sessionId=memory.get("session_id"),
                ownerId=memory.get("user_id"),
                namespace=memory.get("namespace"),
                topics=memory.get("topics") or None,
            ))
        return self._to_plain(self._client.bulk_create_long_term_memories(memories=records))

    def put_working_memory(self, session_id: str, working_memory: dict):
        models = self._models
        responses = []
        now = datetime.now(timezone.utc)
        for message in working_memory.get("messages", []):
            role_name = str(message.get("role") or "system").upper()
            role = getattr(models.MessageRole, role_name, models.MessageRole.SYSTEM)
            original_actor_id = working_memory.get("user_id") or _DEFAULT_USER_ID
            actor_id = _sanitize_actor_id(original_actor_id)
            content_text = str(message.get("content") or "")
            created_at = now
            metadata = {
                "namespace": working_memory.get("namespace"),
                "original_session_id": working_memory.get("original_session_id"),
                "original_actor_id": original_actor_id,
            }
            call_log = {
                "method": "add_session_event",
                "session_id": session_id,
                "actor_id": actor_id,
                "original_actor_id": original_actor_id,
                "role": role_name,
                "content": [_content_log_details(content_text)],
                "created_at": created_at.isoformat(),
                "metadata": metadata,
            }
            logger.info("Redis Agent Memory add_session_event request: %s", _json_log(call_log))
            try:
                response = self._client.add_session_event(
                    session_id=session_id,
                    actor_id=actor_id,
                    role=role,
                    content=[models.Text(text=content_text)],
                    created_at=created_at,
                    metadata=metadata,
                )
            except Exception:
                logger.warning("Redis Agent Memory add_session_event failed; request: %s", _json_log(call_log), exc_info=True)
                raise
            responses.append(self._to_plain(response))
        return {"session_id": session_id, "events": responses}

    def delete_long_term_memories(self, memory_ids: list[str]):
        return self._to_plain(self._client.bulk_delete_long_term_memories(memory_ids=memory_ids))

    def close(self):
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


SEARCH_SCHEMA = {
    "name": "redis_memory_search",
    "description": "Search Redis Agent Memory Server long-term memory using semantic, keyword, or hybrid search.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {"type": "integer", "description": "Maximum results to return, 1 to 25."},
            "search_mode": {"type": "string", "enum": ["semantic", "keyword", "hybrid"], "description": "Search mode override."},
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "redis_memory_remember",
    "description": "Store an explicit long-term memory in Redis Agent Memory Server.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The durable fact, preference, event, or note to remember."},
            "memory_type": {"type": "string", "enum": ["semantic", "episodic", "message"], "description": "Memory type. Defaults to semantic."},
            "topics": {"type": "array", "items": {"type": "string"}, "description": "Optional topics for filtering."},
            "entities": {"type": "array", "items": {"type": "string"}, "description": "Optional entities for filtering."},
        },
        "required": ["content"],
    },
}

FORGET_SCHEMA = {
    "name": "redis_memory_forget",
    "description": "Delete a Redis Agent Memory Server long-term memory by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "The memory ID to delete."},
        },
        "required": ["memory_id"],
    },
}


class RedisAgentMemoryProvider(MemoryProvider):
    """Hermes memory provider backed by Redis Agent Memory Server."""

    def __init__(self, client_factory: Optional[Callable[[dict], Any]] = None):
        self._client_factory = client_factory or (lambda config: _RedisAMSClient(config))
        self._client = None
        self._config: dict[str, Any] = {}
        self._session_id = ""
        self._user_id = _DEFAULT_USER_ID
        self._namespace = _DEFAULT_NAMESPACE
        self._sync_threads: list[threading.Thread] = []

    @property
    def name(self) -> str:
        return "redis-agent-memory"

    def is_available(self, hermes_home: str | None = None) -> bool:
        config = _load_config(hermes_home)
        return bool(config.get("base_url"))

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home")
        self._config = _load_config(hermes_home)
        if not self._config.get("base_url"):
            self._config["base_url"] = _DEFAULT_BASE_URL
        self._session_id = session_id
        identity = str(kwargs.get("agent_identity") or "default")
        raw_namespace = str(self._config.get("namespace") or _DEFAULT_NAMESPACE)
        self._namespace = _sanitize_namespace(raw_namespace.format(identity=identity))
        self._user_id = str(kwargs.get("user_id") or self._config.get("user_id") or _DEFAULT_USER_ID)
        self._config["namespace"] = self._namespace
        self._config["user_id"] = self._user_id
        logged_config = {k: ("<set>" if k == "auth_token" and v else v) for k, v in self._config.items()}
        logger.info("Redis Agent Memory provider initializing: session_id=%s config=%s", session_id, logged_config)
        self._client = self._client_factory(self._config)

    def get_config_schema(self):
        return [
            {
                "key": "base_url",
                "description": "Redis Agent Memory Server base URL",
                "default": _DEFAULT_BASE_URL,
                "env_var": "REDIS_AGENT_MEMORY_URL",
            },
            {
                "key": "auth_token",
                "description": "Redis Agent Memory Server bearer/token auth value, if auth is enabled",
                "secret": True,
                "required": False,
                "env_var": "REDIS_AGENT_MEMORY_TOKEN",
            },
            {"key": "user_id", "description": "Default user identifier", "default": _DEFAULT_USER_ID},
            {"key": "namespace", "description": "Default namespace; supports {identity}", "default": _DEFAULT_NAMESPACE},
            {"key": "store_id", "description": "Redis Cloud Memory store ID", "env_var": "REDIS_AGENT_MEMORY_STORE_ID"},
            {"key": "service_name", "description": "Redis Cloud Memory service name", "env_var": "REDIS_AGENT_MEMORY_SERVICE_NAME"},
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        values = dict(values or {})
        values.pop("auth_token", None)
        values.pop("api_key", None)
        path = Path(hermes_home) / _CONFIG_FILENAME
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    existing = raw
            except Exception:
                existing = {}
        existing.update(values)
        path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def system_prompt_block(self) -> str:
        return (
            "# Redis Agent Memory\n"
            f"Active via Redis Agent Memory Server. User: {self._user_id}. Namespace: {self._namespace}.\n"
            "Use redis_memory_search for relevant recall and redis_memory_remember for explicit durable facts."
        )

    def _search(self, query: str, *, limit: Optional[int] = None, search_mode: Optional[str] = None) -> list[dict]:
        if not self._client or not query:
            return []
        mode = (search_mode or self._config.get("search_mode") or _DEFAULT_SEARCH_MODE).lower()
        if mode not in _VALID_SEARCH_MODES:
            mode = _DEFAULT_SEARCH_MODE
        logger.info(
            "Redis Agent Memory search request: text_chars=%s limit=%s search_mode=%s user_id=%s namespace=%s",
            len(query),
            limit or self._config.get("max_recall_results", 8),
            mode,
            self._user_id,
            self._namespace,
        )
        response = self._client.search_long_term_memory(
            text=query,
            limit=limit or self._config.get("max_recall_results", 8),
            search_mode=mode,
            user_id=self._user_id,
            namespace=self._namespace,
        )
        if isinstance(response, dict):
            memories = response.get("memories") or response.get("results") or response.get("items") or []
        else:
            memories = getattr(response, "memories", []) or getattr(response, "results", []) or getattr(response, "items", []) or []
        normalized = []
        for item in memories:
            if isinstance(item, dict):
                text = item.get("text") or item.get("memory") or item.get("content") or ""
                if text:
                    normalized.append({
                        "id": item.get("id", ""),
                        "text": text,
                        "memory_type": item.get("memory_type", ""),
                        "topics": item.get("topics", []),
                        "score": item.get("score"),
                        "dist": item.get("dist"),
                    })
            else:
                text = getattr(item, "text", "") or getattr(item, "memory", "") or getattr(item, "content", "")
                if text:
                    normalized.append({
                        "id": getattr(item, "id", ""),
                        "text": text,
                        "memory_type": getattr(item, "memory_type", ""),
                        "topics": getattr(item, "topics", []),
                        "score": getattr(item, "score", None),
                        "dist": getattr(item, "dist", None),
                    })
        return normalized

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._config.get("auto_recall", True):
            return ""
        try:
            memories = self._search(query, limit=self._config.get("max_recall_results", 8))
        except Exception as exc:
            logger.debug("Redis Agent Memory prefetch failed: %s", exc)
            return ""
        if not memories:
            return ""
        lines = []
        for memory in memories:
            bits = []
            if memory.get("memory_type"):
                bits.append(str(memory["memory_type"]))
            topics = memory.get("topics") or []
            if topics:
                bits.append(",".join(str(t) for t in topics[:3]))
            prefix = f"[{'; '.join(bits)}] " if bits else ""
            lines.append(f"- {prefix}{memory['text']}")
        intro = "Relevant long-term memories from Redis Agent Memory Server. Use silently when helpful; do not force them into the conversation."
        return "<redis-agent-memory-context>\n" + intro + "\n" + "\n".join(lines) + "\n</redis-agent-memory-context>"

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._client or not self._config.get("auto_sync_turns", True):
            return
        sid = _sanitize_session_id(session_id or self._session_id)
        original_sid = session_id or self._session_id
        payload = {
            "session_id": sid,
            "original_session_id": original_sid,
            "user_id": self._user_id,
            "namespace": self._namespace,
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
        }

        def _sync():
            try:
                logger.info("Redis Agent Memory sync request: %s", _sync_payload_log_details(payload))
                self._client.put_working_memory(sid, payload)
            except Exception as exc:
                logger.warning("Redis Agent Memory sync failed: %s; request: %s", exc, _sync_payload_log_details(payload))

        thread = threading.Thread(target=_sync, daemon=True, name="redis-agent-memory-sync")
        self._sync_threads.append(thread)
        thread.start()

    def _create_memory(self, content: str, *, memory_type: str = "semantic", topics: Optional[list[str]] = None,
                       entities: Optional[list[str]] = None) -> dict:
        if not self._client:
            raise RuntimeError("Redis Agent Memory client is not initialized")
        memory = {
            "text": content.strip(),
            "memory_type": memory_type if memory_type in {"semantic", "episodic", "message"} else "semantic",
            "topics": topics or ["hermes"],
            "entities": entities or [],
            "user_id": self._user_id,
            "namespace": self._namespace,
        }
        logger.info(
            "Redis Agent Memory create_long_term_memory request: text_chars=%s memory_type=%s topics=%s user_id=%s namespace=%s",
            len(memory["text"]),
            memory["memory_type"],
            memory["topics"],
            memory["user_id"],
            memory["namespace"],
        )
        return self._client.create_long_term_memory([memory])

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if action != "add" or not content or not self._client:
            return
        topics = ["hermes", target or "memory"]

        def _sync():
            try:
                self._create_memory(content, memory_type="semantic", topics=topics)
            except Exception as exc:
                logger.warning("Redis Agent Memory mirror failed: %s", exc)

        thread = threading.Thread(target=_sync, daemon=True, name="redis-agent-memory-mirror")
        self._sync_threads.append(thread)
        thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, REMEMBER_SCHEMA, FORGET_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "redis_memory_search":
            query = str(args.get("query") or "").strip()
            if not query:
                return tool_error("Missing required parameter: query")
            limit = _as_int(args.get("limit", self._config.get("max_recall_results", 8)), 8, minimum=1, maximum=25)
            try:
                results = self._search(query, limit=limit, search_mode=args.get("search_mode"))
                return json.dumps({"results": results, "count": len(results)})
            except Exception as exc:
                return tool_error(f"Redis Agent Memory search failed: {exc}")

        if tool_name == "redis_memory_remember":
            content = str(args.get("content") or "").strip()
            if not content:
                return tool_error("Missing required parameter: content")
            topics = args.get("topics") if isinstance(args.get("topics"), list) else ["hermes", "explicit"]
            entities = args.get("entities") if isinstance(args.get("entities"), list) else []
            try:
                response = self._create_memory(
                    content,
                    memory_type=str(args.get("memory_type") or "semantic"),
                    topics=topics,
                    entities=entities,
                )
                return json.dumps({"stored": True, "response": response})
            except Exception as exc:
                return tool_error(f"Redis Agent Memory remember failed: {exc}")

        if tool_name == "redis_memory_forget":
            memory_id = str(args.get("memory_id") or "").strip()
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            try:
                logger.info("Redis Agent Memory delete_long_term_memories request: memory_ids=%s", [memory_id])
                response = self._client.delete_long_term_memories([memory_id])
                if isinstance(response, dict):
                    deleted_value = response.get("deleted", 1)
                    deleted = len(deleted_value) if isinstance(deleted_value, list) else deleted_value
                else:
                    deleted = 1
                return json.dumps({"deleted": deleted, "response": response})
            except Exception as exc:
                return tool_error(f"Redis Agent Memory forget failed: {exc}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for thread in list(self._sync_threads):
            if thread.is_alive():
                thread.join(timeout=5.0)
        self._sync_threads.clear()
        close = getattr(self._client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("Redis Agent Memory client close failed", exc_info=True)


def register(ctx) -> None:
    ctx.register_memory_provider(RedisAgentMemoryProvider())
