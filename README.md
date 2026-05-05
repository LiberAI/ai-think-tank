# 💬 AI Think Tank

Real-time brainstorming between several AI models. Pose a topic with optional
evidence; the app drives multiple OpenRouter models through a three-phase
discussion — **parallel race → closing reflections → consensus wiki** — and
streams the whole thing to a single-page frontend.

## Phase 1 — Parallel race

Each model runs its own independent loop:

- It fires a chat-completion request to OpenRouter on a snapshot of the
  transcript at that moment.
- Whichever model finishes first appends its reply to the shared transcript.
- Slow replies are **not** cancelled — they land later, on top of whatever the
  other models said in the meantime. The `stale_by` field on each message
  records how many messages had appeared between the snapshot and the append.
- After landing a message, a model must wait `ceil(N/2)` other messages before
  it can fire again (the cooldown), where `N` is the number of participants.
- Replies are hard-truncated to `char_limit + 100` characters; the system
  prompt asks for `char_limit` to keep them concise.
- Models are encouraged to address each other with `@Name` tags, which the UI
  highlights in the addressee's color.

## Phase 2 — Closing reflections

Once the race ends, every model that spoke is asked, in parallel:

> *Have you revised your initial position?*

Each reflection starts with a tag of the form `[shift: N]` (0–100, where 0 =
unchanged, 100 = completely revised). The server parses the tag and the UI
renders a small bar in the model's color showing the magnitude of the shift.

## Phase 3 — Consensus wiki

The models then crowdsource a shared summary, wiki-style:

1. A random first editor proposes the initial draft.
2. For each remaining model in random order, that model proposes ONE focused
   revision (an addition, deletion, rephrasing, or correction). The full
   revised article comes back via a `SUMMARY: …\n===ARTICLE===\n<body>`
   delimiter protocol; JSON and bare-prose fallbacks keep parsing robust.
3. The other models vote in parallel: `yes`, `no`, or `abstain` (with a
   one-sentence reason). Votes stream into the UI one at a time.
4. The revision is applied iff `1 + yes_count ≥ N / 2` — the proposer's
   implicit yes plus enough explicit yeses from the rest.
5. The current draft is updated and the next proposer takes a turn.

The UI shows a live diff (line-level, +/− highlighted) for every proposal so
you can see exactly what was added or removed, plus a final "Consensus
summary" panel and a tally of accepted vs. rejected revisions.

The initial draft is capped at `char_limit × 3` characters; revisions from
turn 2 onwards have **no character cap**.

## Persistence

Every discussion is auto-saved as it runs:

- `discussions/{timestamp}-{slug}-{shortid}.md` — full transcript
  (messages + reflections with shifts + consensus turn-by-turn log with
  diffs and votes).
- `discussions/{base}-consensus.md` — companion file with just the final
  consensus summary, written once at the end of phase 3.

Files are refreshed after every message and turn, so a crash mid-run still
leaves a partial transcript on disk. The `discussions/` directory is
gitignored.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then add your OPENROUTER_API_KEY
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
- Model icons live in `frontend/icons/` and are mapped to model ids in
  `MODEL_ICONS` (and colors in `MODEL_COLORS`) inside `frontend/index.html`.

## Tunables (per discussion)

| Field | Meaning | Default |
| --- | --- | --- |
| `topic` | The question or prompt to discuss | required |
| `context` | Free-form evidence / background passed to every model | `""` |
| `char_limit` | Hard cap on the length of each phase-1 reply (chars) | 600 |
| `max_turns` | Total fires across all participants in phase 1 | 12 |
| `models` | List of `{id, name}` from OpenRouter | all defaults |

Cooldown is derived: `min(ceil(N/2), N - 1)`. The wiki article cap on the
initial draft is `char_limit × 3`.

## API

- `GET  /api/health` — `{ ok, openrouter_configured }`
- `GET  /api/models` — default model list
- `POST /api/discussions` — start a discussion; returns `{ id, models, ... }`
- `GET  /api/discussions/{id}` — current state and history
- `GET  /api/discussions/{id}/stream` — SSE stream. Event types:
  - **Phase 1:** `start`, `speaker_started`, `message`, `speaker_failed`
  - **Phase 2:** `closing_started`, `reflection`, `reflection_failed`
  - **Phase 3:** `consensus_started`, `consensus_proposing`,
    `consensus_voting`, `consensus_vote`, `consensus_turn`,
    `consensus_completed`, `consensus_failed`
  - **Terminal:** `completed`, `error`

## Caveats

- Discussions live in memory only; they vanish on restart (transcripts
  remain on disk).
- The initial burst means all `N` models reply once on an empty transcript,
  so the first `N` messages are independent opening statements.
- Every fire bills against OpenRouter — slow models that arrive late still
  cost what they cost.
- Phase 3 makes one proposal call per model plus `(N − 1)` vote calls per
  proposal turn, so total cost roughly scales `O(N²)`.
- If you want stale phase-1 replies dropped instead of appended, add a check
  in `speaker_loop` against `len(disc.history) - snapshot_len` before
  appending.
