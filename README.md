# hermes-redis-agent-memory

Redis Agent Memory Server provider for Hermes Agent.

This project provides a Hermes `MemoryProvider` backed by Redis Agent Memory Server (AMS). It gives Hermes a Redis-native memory backend with both:

- working memory: session-scoped conversation turns and context
- long-term memory: persistent facts, preferences, events, and hybrid search

## Current scope

Implemented provider behavior:

- Hermes memory provider entry point: `register(ctx)`
- config loading from environment and `$HERMES_HOME/redis-agent-memory.json`
- long-term memory prefetch via Redis AMS search
- background `queue_prefetch()` cache for next-turn recall
- working-memory turn sync via Redis AMS session events
- durable SQLite write-behind queue for working-memory sync retries
- initialization health check with fail-open provider disablement
- small circuit breaker for repeated Redis AMS failures
- primary-agent write gating: cron/subagent contexts can recall, but do not write synthetic memory
- session-switch handling with parent-session tracking and prefetch cache invalidation on reset/rewind
- mirroring built-in Hermes memory `add`, `replace`, and `remove` into Redis long-term memory when Hermes provides enough metadata
- provenance topics for mirrored built-in writes, including origin/session/platform when supplied by Hermes
- safe logging: content bodies are not logged, only request shape and character counts

Tools exposed to Hermes:

- `redis_memory_search` — search long-term memory using `semantic`, `keyword`, or `hybrid`
- `redis_memory_remember` — create explicit long-term memories
- `redis_memory_forget` — delete long-term memories by ID with normalized delete counts
- `redis_memory_get` — inspect a long-term memory by ID
- `redis_memory_update` — update a long-term memory by ID; supports text, memory type, and topics
- `redis_memory_status` — inspect provider health, scope, write gating, circuit-breaker state, and pending write queue depth

**Batch tools (added for Hermes expanded memory management tool support):**
- `redis_memory_remember_batch` — create many memories in one call (uses SDK bulk_create)
- `redis_memory_forget_batch` — delete many by ID in one call (uses SDK bulk_delete)
- `redis_memory_update_batch` — update many; provider loops per-ID updates (no bulk_update in current AMS SDK) but still single tool call to agent

## Expected Hermes install layout

Hermes memory-provider discovery maps `memory.provider` to the plugin directory name. Install this plugin under `$HERMES_HOME/plugins/redis-agent-memory/` even though the internal Python package directory is named `hermes_redis_agent_memory` to avoid colliding with the official `redis_agent_memory` SDK.

For development, symlink it:

```bash
mkdir -p ~/.hermes/plugins
ln -s /path/to/hermes-redis-agent-memory/plugins/memory/hermes_redis_agent_memory \
  ~/.hermes/plugins/redis-agent-memory
```

Then install the SDK dependency into Hermes' venv and configure the provider:

```bash
uv pip install --python ~/.hermes/hermes-agent/venv/bin/python redis-agent-memory
hermes config set memory.provider redis-agent-memory
hermes memory status
```

## Environment and deployment modes

This provider always connects Hermes to an **Agent Memory Server HTTP API**. The
provider does not connect directly to Redis on port `6379`; Redis is the storage
engine behind Agent Memory Server.

There are two common deployment modes:

1. **Redis Enterprise Cloud managed memory service** — Hermes points at the
   managed Redis Agent Memory service endpoint. Redis Enterprise Cloud owns the
   Redis database and service lifecycle.
2. **Local Redis plus local Agent Memory Server** — Hermes points at a locally
   running Agent Memory Server, and that server points at a local Redis database.

### Option A: Redis Enterprise Cloud managed service

Use this when Redis Enterprise Cloud is hosting the Redis Agent Memory service.
Hermes only needs the service URL, store ID, and token if the service requires
authentication.

```bash
export REDIS_AGENT_MEMORY_URL=https://your-memory-service.example.com
export REDIS_AGENT_MEMORY_STORE_ID=your-store-id
export REDIS_AGENT_MEMORY_SERVICE_NAME=hermes-memory
# optional, if Redis AMS auth is enabled
export REDIS_AGENT_MEMORY_TOKEN=...
```

Equivalent `$HERMES_HOME/redis-agent-memory.json`:

```json
{
  "base_url": "https://your-memory-service.example.com",
  "store_id": "your-store-id",
  "service_name": "hermes-memory",
  "user_id": "john",
  "namespace": "hermes-{identity}",
  "search_mode": "hybrid",
  "max_recall_results": 8,
  "auto_recall": true,
  "auto_sync_turns": true,
  "api_timeout": 5.0
}
```

### Option B: local Redis plus local Agent Memory Server

Use this for local development or private testing. Hermes still talks to Agent
Memory Server on HTTP port `8000`; Agent Memory Server talks to Redis on port
`6379`.

Example local layout:

- Redis source/install: `/home/john/redis`
- Redis server: `localhost:6379`
- Agent Memory Server source/install: `/home/john/agent-memory-server`
- Agent Memory Server API: `http://localhost:8000`

Hermes provider configuration:

```bash
export REDIS_AGENT_MEMORY_URL=http://localhost:8000
export REDIS_AGENT_MEMORY_STORE_ID=local-hermes-memory
export REDIS_AGENT_MEMORY_SERVICE_NAME=local-hermes-memory
# usually unset for local unauthenticated development
unset REDIS_AGENT_MEMORY_TOKEN
```

Equivalent `$HERMES_HOME/redis-agent-memory.json`:

```json
{
  "base_url": "http://localhost:8000",
  "store_id": "local-hermes-memory",
  "service_name": "local-hermes-memory",
  "user_id": "john",
  "namespace": "hermes-{identity}",
  "search_mode": "hybrid",
  "max_recall_results": 8,
  "auto_recall": true,
  "auto_sync_turns": true,
  "api_timeout": 5.0
}
```

Before starting Hermes, verify both local ports are reachable:

```bash
redis-cli -h localhost -p 6379 PING
curl -fsS http://localhost:8000/health
```

If Redis is reachable but `http://localhost:8000/health` fails, fix/start Agent
Memory Server first. Hermes cannot use the local Redis database directly without
Agent Memory Server running.

SDK-compatible aliases are also accepted:

- `AGENT_MEMORY_API_KEY` as an alias for `REDIS_AGENT_MEMORY_TOKEN`
- `AGENT_MEMORY_STORE_ID` as an alias for `REDIS_AGENT_MEMORY_STORE_ID`

## Operational notes

- The provider writes a local SQLite queue at `$HERMES_HOME/redis-agent-memory-queue.db` for failed working-memory syncs. Pending rows are retried during provider initialization and on later sync attempts.
- Failed health checks disable the Redis provider for the session instead of crashing Hermes.
- After repeated Redis AMS failures, the provider opens a temporary circuit breaker and returns tool errors instead of repeatedly hammering the backend.
- Built-in Hermes memory replacement/removal mirroring depends on metadata from Hermes. If Hermes does not pass the original text for a replace/remove, the provider cannot reliably identify the corresponding Redis memory ID and will fail safe.
- Background worker threads are pruned and joined on shutdown to avoid unbounded thread bookkeeping.

## Development

Run tests without modifying the project or Hermes venv:

```bash
uv run --with pytest pytest tests/ -q
```

The system Python and Hermes venv may not have pytest installed. Prefer the `uv run --with pytest ...` command above unless you deliberately install test dependencies into an environment.

## Batch Memory Operations Support

This release adds support for the expanded batch memory management tools introduced in Hermes (see @Teknium announcement on merged batch save/edit/remove capabilities).

### Why batch tools?
- Reduces agent/tool-call turns dramatically when ingesting or maintaining many facts at once.
- Single tool invocation from the agent instead of N sequential calls.
- Leverages existing SDK bulk endpoints where available.

### New tools

- **`redis_memory_remember_batch`**
  - Input: `{"memories": [ {"content": "...", "memory_type": "semantic", "topics": [...], "entities": [...] }, ... ] }`
  - Uses `_client.create_long_term_memory(list)` → SDK `bulk_create_long_term_memories`
  - Returns aggregated `{"stored": true, "count": N, "response": ...}`

- **`redis_memory_forget_batch`**
  - Input: `{"memory_ids": ["id1", "id2", ...]}`
  - Uses SDK `bulk_delete_long_term_memories`
  - Normalized response includes `deleted` count and the list of IDs.

- **`redis_memory_update_batch`**
  - Input: `{"updates": [ {"memory_id": "id1", "content": "new text", "topics": [...] }, ... ]}`
  - Loops over per-ID `update_long_term_memory` (current AMS SDK has no `bulk_update` endpoint).
  - Returns `{"updated_count": K, "results": [...], "errors": [...], "total": M}`
  - Still provides the single-call win to the calling agent.

### Implementation notes
- Batch create and delete delegate directly to the `redis-agent-memory` SDK bulk methods.
- Batch update falls back to serial per-ID calls inside the provider (still atomic from Hermes perspective).
- All batch tools respect the same config (namespace, user_id, writes_enabled gating, circuit breaker).
- Mutating batch tools (remember/forget/update + _batch) are rejected with an error when `_writes_enabled=False` (cron/subagent contexts).
- Logging and error handling mirror the single-item paths.
- Tests extended with `test_batch_remember_forget_update_tools`.

### Compatibility
- Old single-item tools remain fully supported.
- Providers that implement the batch schemas allow Hermes to prefer batch paths automatically where possible.
- No breaking changes.

See source `get_tool_schemas()` and `handle_tool_call()` for full parameter schemas and dispatch logic.

