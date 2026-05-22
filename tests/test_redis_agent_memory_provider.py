import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugins" / "memory"))
sys.path.insert(0, "/home/john/.hermes/hermes-agent")


class FakeMemoryClient:
    def __init__(self):
        self.search_calls = []
        self.created = []
        self.working_memory_puts = []
        self.deleted = []
        self.fail_working_memory = False

    def search_long_term_memory(self, **kwargs):
        self.search_calls.append(kwargs)
        return {
            "memories": [
                {
                    "id": "m1",
                    "text": "John works at Redis as a Technical Account Manager.",
                    "memory_type": "semantic",
                    "topics": ["work", "redis"],
                    "dist": 0.12,
                },
                {
                    "id": "m2",
                    "text": "John is interested in chess and AI.",
                    "memory_type": "semantic",
                    "topics": ["interests"],
                    "dist": 0.25,
                },
            ]
        }

    def create_long_term_memory(self, memories):
        self.created.extend(memories)
        return {"memory_ids": [f"m{idx}" for idx, _ in enumerate(memories, start=10)]}

    def put_working_memory(self, session_id, working_memory):
        self.working_memory_puts.append((session_id, working_memory))
        if self.fail_working_memory:
            raise RuntimeError("simulated working-memory failure")
        return {"session_id": session_id, "messages": working_memory.get("messages", [])}

    def delete_long_term_memories(self, memory_ids):
        self.deleted.extend(memory_ids)
        return {"deleted": len(memory_ids)}


def test_provider_loads_config_from_env_and_profile_file(tmp_path, monkeypatch):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    (tmp_path / "redis-agent-memory.json").write_text(
        json.dumps({
            "base_url": "http://file-server:8000",
            "namespace": "hermes-{identity}",
            "store_id": "store-from-file",
            "service_name": "service-from-file",
            "max_recall_results": 3,
            "search_mode": "keyword",
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("REDIS_AGENT_MEMORY_URL", "http://env-server:8000")
    monkeypatch.setenv("REDIS_AGENT_MEMORY_TOKEN", "secret-token")
    monkeypatch.setenv("REDIS_AGENT_MEMORY_STORE_ID", "store-from-env")
    monkeypatch.setenv("REDIS_AGENT_MEMORY_SERVICE_NAME", "service-from-env")

    provider = RedisAgentMemoryProvider(client_factory=lambda config: FakeMemoryClient())
    provider.initialize("session-1", hermes_home=str(tmp_path), agent_identity="coder", user_id="john")

    assert provider.name == "redis-agent-memory"
    assert provider._config["base_url"] == "http://file-server:8000"
    assert provider._config["auth_token"] == "secret-token"
    assert provider._config["store_id"] == "store-from-file"
    assert provider._config["service_name"] == "service-from-file"
    assert provider._namespace == "hermes-coder"
    assert provider._user_id == "john"
    assert provider._config["max_recall_results"] == 3
    assert provider._config["search_mode"] == "keyword"


def test_is_available_requires_base_url_not_network(monkeypatch, tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    monkeypatch.delenv("REDIS_AGENT_MEMORY_URL", raising=False)
    provider = RedisAgentMemoryProvider()
    assert provider.is_available(hermes_home=str(tmp_path)) is False

    monkeypatch.setenv("REDIS_AGENT_MEMORY_URL", "http://localhost:8000")
    assert provider.is_available(hermes_home=str(tmp_path)) is True


def test_prefetch_formats_hybrid_long_term_results(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john", agent_identity="default")

    block = provider.prefetch("what do you know about John's work?", session_id="session-1")

    assert "<redis-agent-memory-context>" in block
    assert "John works at Redis" in block
    assert "John is interested in chess" in block
    assert fake.search_calls[0]["text"] == "what do you know about John's work?"
    assert fake.search_calls[0]["search_mode"] == "hybrid"
    assert fake.search_calls[0]["user_id"] == "john"
    assert fake.search_calls[0]["namespace"] == "hermes"


def test_sync_turn_writes_working_memory_with_user_namespace_and_session(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john")

    provider.sync_turn("hello", "hi there", session_id="session-1")
    provider.shutdown()

    assert len(fake.working_memory_puts) == 1
    session_id, payload = fake.working_memory_puts[0]
    assert session_id == "session-1"
    assert payload["user_id"] == "john"
    assert payload["namespace"] == "hermes"
    assert payload["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_sync_turn_sanitizes_hermes_session_ids_for_redis_session_events(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("20260521_232324_03c74e", hermes_home=str(tmp_path), user_id="john")

    provider.sync_turn("hello", "hi there", session_id="20260521_232324_03c74e")
    provider.shutdown()

    session_id, payload = fake.working_memory_puts[0]
    assert session_id == "20260521-232324-03c74e"
    assert payload["session_id"] == "20260521-232324-03c74e"
    assert payload["original_session_id"] == "20260521_232324_03c74e"


def test_sync_turn_logs_request_details_on_failure(tmp_path, caplog):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    fake.fail_working_memory = True
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("20260521_232324_03c74e", hermes_home=str(tmp_path), user_id="john")

    with caplog.at_level("WARNING", logger="hermes_redis_agent_memory.redis_agent_memory"):
        provider.sync_turn("hello", "hi there", session_id="20260521_232324_03c74e")
        provider.shutdown()

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "Redis Agent Memory sync failed" in log_text
    assert "simulated working-memory failure" in log_text
    assert "session_id=20260521-232324-03c74e" in log_text
    assert "original_session_id=20260521_232324_03c74e" in log_text
    assert "user_id=john" in log_text
    assert "namespace=hermes" in log_text
    assert "messages=[user:5 chars, assistant:8 chars]" in log_text


def test_memory_write_mirrors_builtin_user_memory_to_long_term(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john")

    provider.on_memory_write("add", "user", "John prefers concise technical summaries.")
    provider.shutdown()

    assert len(fake.created) == 1
    memory = fake.created[0]
    assert memory["text"] == "John prefers concise technical summaries."
    assert memory["memory_type"] == "semantic"
    assert memory["topics"] == ["hermes", "user"]
    assert memory["user_id"] == "john"
    assert memory["namespace"] == "hermes"


def test_tool_schemas_and_tool_calls_search_remember_forget(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john")

    tool_names = {schema["name"] for schema in provider.get_tool_schemas()}
    assert {"redis_memory_search", "redis_memory_remember", "redis_memory_forget"}.issubset(tool_names)

    search_result = json.loads(provider.handle_tool_call("redis_memory_search", {"query": "redis", "limit": 2}))
    assert search_result["count"] == 2
    assert search_result["results"][0]["text"].startswith("John works at Redis")

    remember_result = json.loads(provider.handle_tool_call("redis_memory_remember", {"content": "John likes chess."}))
    assert remember_result["stored"] is True
    assert fake.created[-1]["text"] == "John likes chess."

    forget_result = json.loads(provider.handle_tool_call("redis_memory_forget", {"memory_id": "m1"}))
    assert forget_result["deleted"] == 1
    assert fake.deleted == ["m1"]
