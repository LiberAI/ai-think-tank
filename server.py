"""AI Think Tank — parallel-race orchestration over OpenRouter models."""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CHAR_LIMIT = 600
DEFAULT_MAX_TURNS = 12

DEFAULT_MODELS = [
    {"id": "anthropic/claude-sonnet-4.5", "name": "Claude"},
    {"id": "openai/gpt-4o", "name": "GPT-4o"},
    {"id": "google/gemini-2.5-pro", "name": "Gemini"},
    {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama"},
    {"id": "mistralai/mistral-large", "name": "Mistral"},
    {"id": "deepseek/deepseek-chat", "name": "DeepSeek"},
]


@dataclass
class ModelSpec:
    id: str
    name: str


@dataclass
class Message:
    turn: int
    model_id: str
    model_name: str
    content: str
    elapsed_ms: int


@dataclass
class Discussion:
    id: str
    topic: str
    context: str
    char_limit: int
    max_turns: int
    models: list
    history: list = field(default_factory=list)
    last_spoke: dict = field(default_factory=dict)
    status: str = "pending"
    events: list = field(default_factory=list)
    completed: bool = False
    notify: asyncio.Event = field(default_factory=asyncio.Event)


DISCUSSIONS: dict[str, Discussion] = {}


def emit(disc: Discussion, event: dict) -> None:
    disc.events.append(event)
    old, disc.notify = disc.notify, asyncio.Event()
    old.set()


def build_system_prompt(disc: Discussion, model: ModelSpec) -> str:
    return (
        f"You are {model.name}, one of several AI models taking part in a real-time think-tank "
        f"discussion. You will see the running transcript and reply with your next contribution. "
        f"Build on, challenge, refine, or question prior points; introduce fresh angles when useful. "
        f"Be concise, substantive and conversational.\n\n"
        f"STRICT LIMIT: keep your reply under {disc.char_limit} characters. "
        f"Do not preface with your own name or any role tag — your message will be attributed automatically.\n\n"
        f"TOPIC:\n{disc.topic}\n\n"
        f"EVIDENCE / CONTEXT:\n{disc.context or '(none provided)'}"
    )


def build_messages(disc: Discussion, model: ModelSpec, history: list) -> list[dict]:
    msgs = [{"role": "system", "content": build_system_prompt(disc, model)}]
    if not history:
        msgs.append({
            "role": "user",
            "content": "You are opening the discussion. Share your initial perspective.",
        })
        return msgs
    transcript = "\n\n".join(
        f"{('You' if m.model_id == model.id else m.model_name)}: {m.content}"
        for m in history
    )
    msgs.append({
        "role": "user",
        "content": (
            f"Discussion transcript so far:\n\n{transcript}\n\n"
            f"Your next reply (under {disc.char_limit} chars, no name prefix):"
        ),
    })
    return msgs


def truncate(content: str, char_limit: int) -> str:
    content = (content or "").strip()
    if len(content) <= char_limit:
        return content
    cut = content[:char_limit].rstrip()
    if " " in cut[-40:]:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


async def call_model(client: httpx.AsyncClient, disc: Discussion, model: ModelSpec, history: list) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_REFERRER", "http://localhost:8000"),
        "X-Title": "AI Think Tank",
    }
    body = {
        "model": model.id,
        "messages": build_messages(disc, model, history),
        "max_tokens": max(256, disc.char_limit),
        "temperature": 0.85,
    }
    r = await client.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        json=body,
        headers=headers,
        timeout=120.0,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    return truncate(data["choices"][0]["message"]["content"], disc.char_limit)


async def run_discussion(disc: Discussion) -> None:
    """Each model runs its own loop, firing as soon as it's free and off cooldown.

    Slow responses are NOT cancelled - they land on the transcript whenever they
    return, even if other models have spoken in the meantime (so they're replying
    to a stale snapshot). After landing a message a model must wait `cooldown`
    other messages before firing again.
    """
    disc.status = "running"
    n = len(disc.models)
    cooldown = math.ceil(n / 2) if n > 1 else 0
    cooldown = min(cooldown, n - 1)

    emit(disc, {
        "type": "start",
        "topic": disc.topic,
        "context": disc.context,
        "char_limit": disc.char_limit,
        "max_turns": disc.max_turns,
        "cooldown": cooldown,
        "models": [{"id": m.id, "name": m.name} for m in disc.models],
    })

    fired = 0  # total requests fired (caps at max_turns)
    notify = asyncio.Event()  # fires when transcript grows (woken by announce())
    MAX_CONSECUTIVE_ERRORS = 5

    def announce():
        nonlocal notify
        old, notify = notify, asyncio.Event()
        old.set()

    async def speaker_loop(client: httpx.AsyncClient, model: ModelSpec):
        nonlocal fired
        errors_in_a_row = 0
        while fired < disc.max_turns and errors_in_a_row < MAX_CONSECUTIVE_ERRORS:
            last = disc.last_spoke.get(model.id)
            current_len = len(disc.history)
            # Cooldown: wait until `cooldown` other messages have appeared since I spoke.
            if last is not None and (current_len - last) <= cooldown:
                ev = notify
                await ev.wait()
                continue
            if fired >= disc.max_turns:
                return

            fired += 1
            snapshot_len = len(disc.history)
            snapshot = list(disc.history)
            emit(disc, {
                "type": "speaker_started",
                "model_id": model.id,
                "model_name": model.name,
                "history_len_at_start": snapshot_len,
            })
            started = time.monotonic()
            try:
                content = await call_model(client, disc, model, snapshot)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                errors_in_a_row += 1
                # Release the slot so other speakers can still reach max_turns successes.
                fired -= 1
                emit(disc, {
                    "type": "speaker_failed",
                    "model_id": model.id,
                    "model_name": model.name,
                    "error": str(e),
                })
                # Backoff on repeated failures so we don't hammer a broken model.
                await asyncio.sleep(min(2 ** errors_in_a_row, 30))
                continue

            errors_in_a_row = 0
            elapsed_ms = int((time.monotonic() - started) * 1000)
            msg = Message(
                turn=len(disc.history) + 1,
                model_id=model.id,
                model_name=model.name,
                content=content,
                elapsed_ms=elapsed_ms,
            )
            disc.history.append(msg)
            disc.last_spoke[model.id] = len(disc.history) - 1
            emit(disc, {
                "type": "message",
                "message": asdict(msg),
                "history_len_at_start": snapshot_len,
                "stale_by": (len(disc.history) - 1) - snapshot_len,
            })
            announce()

    try:
        async with httpx.AsyncClient() as client:
            tasks = [asyncio.create_task(speaker_loop(client, m)) for m in disc.models]
            await asyncio.gather(*tasks, return_exceptions=True)
        disc.status = "completed"
        emit(disc, {"type": "completed"})
    except Exception as e:
        disc.status = "error"
        emit(disc, {"type": "error", "message": str(e)})
    finally:
        disc.completed = True
        old, disc.notify = disc.notify, asyncio.Event()
        old.set()


# === FastAPI app ===

app = FastAPI(title="AI Think Tank")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelInput(BaseModel):
    id: str
    name: str


class StartRequest(BaseModel):
    topic: str
    context: str = ""
    char_limit: int = DEFAULT_CHAR_LIMIT
    max_turns: int = DEFAULT_MAX_TURNS
    models: Optional[list[ModelInput]] = None


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "openrouter_configured": bool(os.environ.get("OPENROUTER_API_KEY")),
    }


@app.get("/api/models")
def list_models():
    return DEFAULT_MODELS


@app.post("/api/discussions")
async def start_discussion(req: StartRequest):
    if not req.topic.strip():
        raise HTTPException(400, "Topic is required.")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise HTTPException(500, "OPENROUTER_API_KEY environment variable is not set.")
    selected = req.models or [ModelInput(**m) for m in DEFAULT_MODELS]
    if len(selected) < 2:
        raise HTTPException(400, "At least 2 models are required.")
    models = [ModelSpec(id=m.id, name=m.name) for m in selected]
    disc = Discussion(
        id=str(uuid.uuid4()),
        topic=req.topic.strip(),
        context=req.context.strip(),
        char_limit=max(80, min(req.char_limit, 4000)),
        max_turns=max(1, min(req.max_turns, 100)),
        models=models,
    )
    DISCUSSIONS[disc.id] = disc
    asyncio.create_task(run_discussion(disc))
    return {
        "id": disc.id,
        "models": [{"id": m.id, "name": m.name} for m in models],
        "char_limit": disc.char_limit,
        "max_turns": disc.max_turns,
        "cooldown": min(math.ceil(len(models) / 2), len(models) - 1),
    }


@app.get("/api/discussions/{disc_id}")
def get_discussion(disc_id: str):
    disc = DISCUSSIONS.get(disc_id)
    if not disc:
        raise HTTPException(404, "Unknown discussion")
    return {
        "id": disc.id,
        "topic": disc.topic,
        "status": disc.status,
        "models": [{"id": m.id, "name": m.name} for m in disc.models],
        "history": [asdict(m) for m in disc.history],
    }


@app.get("/api/discussions/{disc_id}/stream")
async def stream(disc_id: str):
    disc = DISCUSSIONS.get(disc_id)
    if not disc:
        raise HTTPException(404, "Unknown discussion")

    async def gen():
        idx = 0
        try:
            while True:
                notify = disc.notify
                while idx < len(disc.events):
                    yield f"data: {json.dumps(disc.events[idx])}\n\n"
                    idx += 1
                if disc.completed and idx >= len(disc.events):
                    return
                try:
                    await asyncio.wait_for(notify.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            return

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


# === Static frontend ===

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


if FRONTEND.is_dir():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
