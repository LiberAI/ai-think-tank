# AI Think Tank

Real-time brainstorming between several AI models. Pose a topic with optional
evidence; the app fires requests in parallel to multiple OpenRouter models and
streams the resulting discussion to a single-page frontend.

## How it works

Each model runs its own independent loop:

- It fires a chat-completion request to OpenRouter on a snapshot of the
  transcript at that moment.
- Whichever model finishes first appends its reply to the shared transcript.
- Slow replies are **not** cancelled — they land later, on top of whatever the
  other models said in the meantime. The `stale_by` field on each message
  records how many messages had appeared between the snapshot and the append.
- After landing a message, a model must wait `ceil(N/2)` other messages before
  it can fire again (the cooldown), where `N` is the number of participants.
- Replies are hard-truncated to a per-discussion character limit (the system
  prompt also asks the model to respect it).

Events are pushed to the browser over Server-Sent Events.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env  # then add your OPENROUTER_API_KEY
```

Get an OpenRouter key at https://openrouter.ai/keys.

## Run

```bash
.venv/bin/uvicorn server:app --reload
```

Open http://localhost:8000.

## Configuration

- `OPENROUTER_API_KEY` (required) — set in `.env` or the environment.
- `OPENROUTER_REFERRER` (optional) — sent as the `HTTP-Referer` header to
  OpenRouter.
- Default model list lives in `DEFAULT_MODELS` in `server.py`. Override per
  request from the UI.

## Tunables (per discussion)

| Field | Meaning | Default |
| --- | --- | --- |
| `topic` | The question or prompt to discuss | required |
| `context` | Free-form evidence / background passed to every model | `""` |
| `char_limit` | Hard cap on the length of each reply (chars) | 600 |
| `max_turns` | Total number of fires across all participants | 12 |
| `models` | List of `{id, name}` from OpenRouter | all defaults |

Cooldown is derived: `min(ceil(N/2), N - 1)`.

## API

- `GET  /api/health` — `{ ok, openrouter_configured }`
- `GET  /api/models` — default model list
- `POST /api/discussions` — start a discussion; returns `{ id, models, ... }`
- `GET  /api/discussions/{id}` — current state and history
- `GET  /api/discussions/{id}/stream` — SSE stream of events:
  `start`, `speaker_started`, `message`, `speaker_failed`, `completed`, `error`

## Caveats

- Discussions live in memory only; they vanish on restart.
- The initial burst means all `N` models reply once on an empty transcript,
  so the first `N` messages are independent opening statements.
- Every fire bills against OpenRouter — slow models that arrive late still
  cost what they cost.
- If you want stale replies dropped instead of appended, add a check in
  `speaker_loop` against `len(disc.history) - snapshot_len` before appending.
