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

## Environment

```bash
export REDIS_AGENT_MEMORY_URL=https://your-memory-service.example.com
export REDIS_AGENT_MEMORY_STORE_ID=your-store-id
export REDIS_AGENT_MEMORY_SERVICE_NAME=hermes-memory
# optional, if Redis AMS auth is enabled
export REDIS_AGENT_MEMORY_TOKEN=...
```

SDK-compatible aliases are also accepted:

- `AGENT_MEMORY_API_KEY` as an alias for `REDIS_AGENT_MEMORY_TOKEN`
- `AGENT_MEMORY_STORE_ID` as an alias for `REDIS_AGENT_MEMORY_STORE_ID`

Optional `$HERMES_HOME/redis-agent-memory.json`:

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
