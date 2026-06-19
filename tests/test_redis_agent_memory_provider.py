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
        self.fail_search = False
        self.fail_health = False

    def search_long_term_memory(self, **kwargs):
        self.search_calls.append(kwargs)
        if self.fail_search:
            raise RuntimeError("simulated search failure")
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
        return {"created": [f"m{idx}" for idx, _ in enumerate(memories, start=10)]}

    def put_working_memory(self, session_id, working_memory):
        self.working_memory_puts.append((session_id, working_memory))
        if self.fail_working_memory:
            raise RuntimeError("simulated working-memory failure")
        return {"session_id": session_id, "messages": working_memory.get("messages", [])}

    def delete_long_term_memories(self, memory_ids):
        self.deleted.extend(memory_ids)
        return {"deleted": len(memory_ids)}

    def health(self):
        if self.fail_health:
            raise RuntimeError("simulated health failure")
        return {"status": "ok"}

    def get_long_term_memory(self, memory_id):
        return {"id": memory_id, "text": "details"}

    def update_long_term_memory(self, memory_id, **kwargs):
        self.updated = getattr(self, "updated", [])
        self.updated.append((memory_id, kwargs))
        return {"id": memory_id, "updated": True, **kwargs}



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
    assert {"redis_memory_remember_batch", "redis_memory_forget_batch", "redis_memory_update_batch"}.issubset(tool_names)

    search_result = json.loads(provider.handle_tool_call("redis_memory_search", {"query": "redis", "limit": 2}))
    assert search_result["count"] == 2
    assert search_result["results"][0]["text"].startswith("John works at Redis")

    remember_result = json.loads(provider.handle_tool_call("redis_memory_remember", {"content": "John likes chess."}))
    assert remember_result["stored"] is True
    assert fake.created[-1]["text"] == "John likes chess."

    forget_result = json.loads(provider.handle_tool_call("redis_memory_forget", {"memory_id": "m1"}))
    assert forget_result["deleted"] == 1
    assert fake.deleted == ["m1"]


def test_non_primary_agent_context_suppresses_writes_but_allows_recall(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("cron-session", hermes_home=str(tmp_path), user_id="john", agent_context="cron", platform="cron")

    assert "John works at Redis" in provider.prefetch("redis", session_id="cron-session")
    provider.sync_turn("cron prompt", "cron response", session_id="cron-session")
    provider.on_memory_write("add", "user", "Cron-only synthetic fact")
    provider.shutdown()

    assert fake.working_memory_puts == []
    assert fake.created == []


def test_on_session_switch_updates_cached_session_id_and_clears_prefetch(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("old_session", hermes_home=str(tmp_path), user_id="john")
    provider._prefetch_results["old_session"] = "stale"

    provider.on_session_switch("new_session", parent_session_id="old_session", reset=True)
    provider.sync_turn("hello", "hi", session_id="")
    provider.shutdown()

    assert provider._session_id == "new_session"
    assert provider._prefetch_results == {}
    session_id, payload = fake.working_memory_puts[0]
    assert session_id == "new-session"
    assert payload["original_session_id"] == "new_session"
    assert payload["parent_session_id"] == "old_session"


def test_sync_turn_accepts_messages_keyword(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john")

    provider.sync_turn("hello", "hi", session_id="session-1", messages=[{"role": "user", "content": "hello"}])
    provider.shutdown()

    assert len(fake.working_memory_puts) == 1


def test_search_mode_is_forwarded_to_client(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john")

    provider.handle_tool_call("redis_memory_search", {"query": "redis", "search_mode": "keyword"})

    assert fake.search_calls[-1]["search_mode"] == "keyword"


def test_initialize_disables_client_when_health_check_fails(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    fake.fail_health = True
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)

    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john")

    assert provider._client is None
    assert provider._last_health_error == "simulated health failure"


def test_queue_prefetch_caches_next_turn_result(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john")

    provider.queue_prefetch("redis", session_id="session-1")
    provider.shutdown()
    block = provider.prefetch("ignored", session_id="session-1")

    assert "John works at Redis" in block
    assert fake.search_calls[0]["text"] == "redis"


def test_circuit_breaker_opens_after_repeated_search_failures(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    fake.fail_search = True
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john")

    for _ in range(3):
        provider.handle_tool_call("redis_memory_search", {"query": "redis"})
    before = len(fake.search_calls)
    result = json.loads(provider.handle_tool_call("redis_memory_search", {"query": "redis"}))

    assert len(fake.search_calls) == before
    assert "temporarily unavailable" in result["error"]


def test_failed_working_memory_writes_are_replayed_from_durable_queue(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    failing = FakeMemoryClient()
    failing.fail_working_memory = True
    provider = RedisAgentMemoryProvider(client_factory=lambda config: failing)
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john")
    provider.sync_turn("hello", "hi", session_id="session-1")
    provider.shutdown()

    assert failing.working_memory_puts == [("session-1", failing.working_memory_puts[0][1])]

    succeeding = FakeMemoryClient()
    provider2 = RedisAgentMemoryProvider(client_factory=lambda config: succeeding)
    provider2.initialize("session-1", hermes_home=str(tmp_path), user_id="john")
    provider2.shutdown()

    assert len(succeeding.working_memory_puts) == 1
    session_id, payload = succeeding.working_memory_puts[0]
    assert session_id == "session-1"
    assert payload["messages"][0]["content"] == "hello"


def test_status_tool_reports_scope_health_and_queue(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john")

    result = json.loads(provider.handle_tool_call("redis_memory_status", {}))

    assert result["provider"] == "redis-agent-memory"
    assert result["configured"] is True
    assert result["healthy"] is True
    assert result["user_id"] == "john"
    assert result["namespace"] == "hermes"
    assert result["pending_writes"] == 0


def test_memory_write_add_preserves_provenance_topics_and_mapping(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john")

    provider.on_memory_write(
        "add",
        "user",
        "John prefers concise answers.",
        metadata={"write_origin": "tool", "session_id": "session-1", "platform": "cli"},
    )
    provider.shutdown()

    memory = fake.created[-1]
    assert memory["topics"] == ["hermes", "user", "origin:tool", "session:session-1", "platform:cli"]
    assert provider._lookup_mirrored_memory_id("user", "John prefers concise answers.") == "m10"


def test_memory_write_replace_deletes_old_mapping_and_creates_new_memory(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john")
    provider.on_memory_write("add", "user", "old fact")
    provider.shutdown()

    provider.on_memory_write("replace", "user", "new fact", metadata={"old_text": "old fact"})
    provider.shutdown()

    assert "m10" in fake.deleted
    assert fake.created[-1]["text"] == "new fact"
    assert provider._lookup_mirrored_memory_id("user", "old fact") == ""
    assert provider._lookup_mirrored_memory_id("user", "new fact") == "m10"


def test_memory_write_remove_deletes_mapped_memory(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john")
    provider.on_memory_write("add", "memory", "temporary fact")
    provider.shutdown()

    provider.on_memory_write("remove", "memory", "temporary fact")
    provider.shutdown()

    assert "m10" in fake.deleted
    assert provider._lookup_mirrored_memory_id("memory", "temporary fact") == ""


def test_forget_tool_normalizes_list_delete_responses(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    class ListDeleteClient(FakeMemoryClient):
        def delete_long_term_memories(self, memory_ids):
            self.deleted.extend(memory_ids)
            return {"deleted": memory_ids}

    fake = ListDeleteClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john")

    result = json.loads(provider.handle_tool_call("redis_memory_forget", {"memory_id": "m1"}))

    assert result["deleted"] == 1
    assert result["memory_ids"] == ["m1"]


def test_get_and_update_by_id_tools(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john")

    got = json.loads(provider.handle_tool_call("redis_memory_get", {"memory_id": "m1"}))
    updated = json.loads(provider.handle_tool_call(
        "redis_memory_update",
        {"memory_id": "m1", "content": "updated details", "topics": ["hermes", "edited"]},
    ))

    assert got["memory"]["text"] == "details"
    assert fake.updated == [("m1", {"text": "updated details", "topics": ["hermes", "edited"]})]
    assert updated["updated"] is True


def test_completed_threads_are_pruned_before_starting_more(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john")

    for idx in range(5):
        provider.on_memory_write("add", "memory", f"fact {idx}")
        provider.shutdown()

    provider.on_memory_write("add", "memory", "fact 6")
    provider.shutdown()

    assert len(provider._sync_threads) == 0


def test_batch_remember_forget_update_tools(tmp_path):
    from hermes_redis_agent_memory import RedisAgentMemoryProvider

    fake = FakeMemoryClient()
    provider = RedisAgentMemoryProvider(client_factory=lambda config: fake)
    provider.initialize("session-1", hermes_home=str(tmp_path), user_id="john")

    # Batch remember
    remember_batch = json.loads(provider.handle_tool_call(
        "redis_memory_remember_batch",
        {"memories": [
            {"content": "Batch fact 1 about chess."},
            {"content": "Batch fact 2 about Redis.", "topics": ["work", "redis"]},
        ]}
    ))
    assert remember_batch.get("stored") is True or "count" in remember_batch
    assert len(fake.created) >= 2

    # Batch forget
    forget_batch = json.loads(provider.handle_tool_call(
        "redis_memory_forget_batch",
        {"memory_ids": ["m1", "m2"]}
    ))
    assert forget_batch["deleted"] == 2 or forget_batch.get("deleted") >= 1
    assert "m1" in fake.deleted and "m2" in fake.deleted

    # Batch update
    update_batch = json.loads(provider.handle_tool_call(
        "redis_memory_update_batch",
        {"updates": [
            {"memory_id": "m10", "content": "Updated batch fact"},
            {"memory_id": "m11", "topics": ["updated", "batch"]},
        ]}
    ))
    assert update_batch["updated_count"] >= 1
    assert len(update_batch.get("results", [])) + len(update_batch.get("errors", [])) > 0

    print("Batch tools test passed")
