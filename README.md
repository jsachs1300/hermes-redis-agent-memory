# hermes-redis-agent-memory

Redis Agent Memory Server provider for Hermes Agent.

This project prototypes a Hermes `MemoryProvider` backed by Redis Agent Memory Server (AMS). The goal is to give Hermes a Redis-native, self-hostable memory backend with both:

- working memory: session-scoped conversation turns and context
- long-term memory: persistent facts, preferences, events, and hybrid search

## Current scope

Implemented in this first pass:

- Hermes memory provider entry point: `register(ctx)`
- config loading from environment and `$HERMES_HOME/redis-agent-memory.json`
- long-term memory prefetch via Redis AMS search
- working-memory turn sync via Redis AMS working memory
- mirroring built-in Hermes memory writes into long-term memory
- tools:
  - `redis_memory_search`
  - `redis_memory_remember`
  - `redis_memory_forget`

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
export REDIS_AGENT_MEMORY_URL=https://gcp-us-east4.memory.redis.io
export REDIS_AGENT_MEMORY_STORE_ID=36df4414091043fb8e1006d8c17b4d9a
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
  "base_url": "https://gcp-us-east4.memory.redis.io",
  "store_id": "36df4414091043fb8e1006d8c17b4d9a",
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

## Development

Run tests with the Hermes venv:

```bash
/home/john/.hermes/hermes-agent/venv/bin/python -m pytest tests/ -q
```
