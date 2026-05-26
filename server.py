"""AI Think Tank — parallel-race orchestration over OpenRouter models."""
from __future__ import annotations

import asyncio
import datetime
import difflib
import json
import math
import os
import random
import re
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
DISPLAY_SLACK = 100

DEFAULT_MODELS = [
    {"id": "openai/gpt-5.1", "name": "GPT-5.1"},
    {"id": "google/gemini-3-flash-preview", "name": "Gemini"},
    {"id": "perplexity/sonar", "name": "Sonar"},
    {"id": "x-ai/grok-4.3", "name": "Grok"},
    {"id": "deepseek/deepseek-v3.2", "name": "DeepSeek"},
    {"id": "anthropic/claude-sonnet-4.5", "name": "Claude"},
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
    shift: Optional[int] = None


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
    started_at: Optional[str] = None
    path: Optional[Path] = None
    reflections: list = field(default_factory=list)
    consensus: list = field(default_factory=list)
    consensus_draft: str = ""


DISCUSSIONS: dict[str, Discussion] = {}
DISCUSSIONS_DIR = Path(__file__).resolve().parent / "discussions"


def slugify(s: str, n: int = 50) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:n] or "topic"


def render_transcript(disc: Discussion) -> str:
    lines = [f"# {disc.topic}", ""]
    lines.append(f"- **Started:** {disc.started_at or '—'}")
    lines.append(f"- **Models:** " + ", ".join(f"{m.name} (`{m.id}`)" for m in disc.models))
    lines.append(f"- **Char limit:** {disc.char_limit} · **Max turns:** {disc.max_turns}")
    lines.append(f"- **Status:** {disc.status}")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(disc.context or "_(none)_")
    lines.append("")
    lines.append("## Transcript")
    lines.append("")
    for m in disc.history:
        lines.append(f"### #{m.turn} — {m.model_name} · {m.elapsed_ms} ms")
        lines.append("")
        lines.append(m.content)
        lines.append("")
    if disc.reflections:
        lines.append("## Closing reflections — Have you revised your initial position?")
        lines.append("")
        for r in disc.reflections:
            shift_str = f" · shift {r.shift}/100" if r.shift is not None else ""
            lines.append(f"### {r.model_name} · {r.elapsed_ms} ms{shift_str}")
            lines.append("")
            lines.append(r.content)
            lines.append("")
    if disc.consensus_draft:
        lines.append("## Consensus wiki (final)")
        lines.append("")
        lines.append(disc.consensus_draft)
        lines.append("")
    if disc.consensus:
        lines.append("## Consensus wiki — turn log")
        lines.append("")
        for t in disc.consensus:
            verdict = "DRAFT" if t["kind"] == "draft" else ("ACCEPTED" if t["accepted"] else "REJECTED")
            tally = (
                "" if t["kind"] == "draft"
                else f" · yes {t['yes']} / no {t['no']} / abstain {t['abstain']}"
            )
            lines.append(f"### Turn {t['turn']} — {t['proposer_name']} · {verdict}{tally}")
            lines.append("")
            lines.append(f"_{t['summary']}_")
            lines.append("")
            if t["kind"] != "draft" and t.get("diff"):
                lines.append("```diff")
                for h in t["diff"]:
                    prefix = " " if h["op"] == "=" else h["op"]
                    lines.append(f"{prefix} {h['text']}")
                lines.append("```")
                lines.append("")
            if t["kind"] != "draft" and t["votes"]:
                for v in t["votes"]:
                    reason = f" — {v['reason']}" if v.get("reason") else ""
                    lines.append(f"- **{v['model_name']}**: {v['vote']}{reason}")
                lines.append("")
            if t["kind"] != "draft" and not t["accepted"] and t["revised"]:
                lines.append("Proposed (rejected) full text:")
                lines.append("")
                lines.append("```")
                lines.append(t["revised"])
                lines.append("```")
                lines.append("")
    return "\n".join(lines)


def save_transcript(disc: Discussion) -> None:
    DISCUSSIONS_DIR.mkdir(exist_ok=True)
    if disc.path is None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        disc.path = DISCUSSIONS_DIR / f"{ts}-{slugify(disc.topic)}-{disc.id[:8]}.md"
    disc.path.write_text(render_transcript(disc), encoding="utf-8")


def save_final_consensus(disc: Discussion) -> Optional[Path]:
    if not disc.consensus_draft:
        return None
    DISCUSSIONS_DIR.mkdir(exist_ok=True)
    if disc.path is not None:
        base = disc.path.stem
    else:
        ts = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        base = f"{ts}-{slugify(disc.topic)}-{disc.id[:8]}"
    target = DISCUSSIONS_DIR / f"{base}-consensus.md"
    body = (
        f"# Consensus summary — {disc.topic}\n\n"
        f"_Discussion started {disc.started_at or '—'} · "
        f"{len(disc.history)} message(s) · {len(disc.consensus)} consensus turn(s) · "
        f"{len(disc.consensus_draft)} characters._\n\n"
        f"{disc.consensus_draft}\n"
    )
    target.write_text(body, encoding="utf-8")
    return target


def emit(disc: Discussion, event: dict) -> None:
    disc.events.append(event)
    old, disc.notify = disc.notify, asyncio.Event()
    old.set()


def build_system_prompt(disc: Discussion, model: ModelSpec) -> str:
    others = [m.name for m in disc.models if m.id != model.id]
    others_list = ", ".join(others) if others else "(none)"
    return (
        f"You are {model.name}, one of several AI models taking part in a real-time think-tank "
        f"discussion. The other participants are: {others_list}. "
        f"You will see the running transcript and reply with your next contribution. "
        f"Build on, challenge, refine, or question prior points; introduce fresh angles when useful. "
        f"When you address or directly respond to another participant, prefix their name with `@` "
        f"(e.g., `@{others[0] if others else 'Name'} makes a good point about X`). "
        f"Use the names exactly as listed above. "
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


SHIFT_RE = re.compile(r"^\s*\[\s*shift\s*:\s*(\d{1,3})\s*\]\s*", re.IGNORECASE)


def extract_shift(content: str) -> tuple[Optional[int], str]:
    m = SHIFT_RE.match(content or "")
    if not m:
        return None, content
    n = max(0, min(100, int(m.group(1))))
    return n, content[m.end():].lstrip()


def parse_proposal(raw: str) -> tuple[str, str]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:\w+)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()
    m = re.match(
        r"\s*summary\s*:\s*(.+?)\n+\s*={3,}\s*article\s*={3,}\s*\n+(.+)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.match(
        r"\s*summary\s*:\s*(.+?)\n+\s*-{3,}\s*\n+(.+)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    parsed = extract_json(text)
    if parsed and parsed.get("revised"):
        return (str(parsed.get("summary") or "(revision)").strip(),
                str(parsed["revised"]).strip())
    if text:
        return "(revision; no summary)", text
    raise RuntimeError("empty proposal")


def parse_vote(raw: str) -> tuple[str, str]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:\w+)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()
    parsed = extract_json(text)
    if isinstance(parsed, dict):
        v = str(parsed.get("vote", "")).strip().lower()
        if v in ("yes", "no", "abstain"):
            return v, (parsed.get("reason") or "").strip()
    m = re.search(r"^\s*vote\s*:\s*(yes|no|abstain)\b", text, re.IGNORECASE | re.MULTILINE)
    if m:
        v = m.group(1).lower()
        rm = re.search(r"^\s*reason\s*:\s*(.+?)$", text, re.IGNORECASE | re.MULTILINE)
        return v, (rm.group(1).strip() if rm else "")
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m2 = re.match(r"(?:i\s+)?vote\s+(yes|no|abstain)\b", s, re.IGNORECASE)
        if m2:
            return m2.group(1).lower(), ""
        for v in ("yes", "no", "abstain"):
            if re.match(rf"^{v}\b", s, re.IGNORECASE):
                return v, s[len(v):].lstrip(" .,:;-").strip()
        break
    return "abstain", "(could not parse vote)"


def extract_json(s: str) -> Optional[dict]:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s).strip()
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            v = json.loads(s[start:end + 1])
            return v if isinstance(v, dict) else None
        except Exception:
            return None
    return None


def build_closing_messages(disc: Discussion, model: ModelSpec) -> list[dict]:
    msgs = [{"role": "system", "content": build_system_prompt(disc, model)}]
    transcript = "\n\n".join(
        f"{('You' if m.model_id == model.id else m.model_name)}: {m.content}"
        for m in disc.history
    )
    msgs.append({
        "role": "user",
        "content": (
            f"The discussion has now concluded. Here is the full transcript:\n\n{transcript}\n\n"
            f"Reviewing the full discussion, have you revised your initial position?\n\n"
            f"Start your reply with a tag of the form `[shift: N]` where N is an integer 0–100 "
            f"(0 = my view is unchanged, 100 = my view has completely flipped). After the tag, "
            f"briefly explain what (if anything) shifted and why; if a participant's argument "
            f"moved you, credit them with `@Name`. Keep your reflection (excluding the tag) under "
            f"{disc.char_limit} characters. No name prefix on your reply."
        ),
    })
    return msgs


async def post_chat(
    client: httpx.AsyncClient,
    disc: Discussion,
    model: ModelSpec,
    messages: list,
    *,
    max_chars: int = 0,
    max_tokens: Optional[int] = None,
) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_REFERRER", "http://localhost:8000"),
        "X-Title": "AI Think Tank",
    }
    body = {
        "model": model.id,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else max(256, disc.char_limit),
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
    content = data["choices"][0]["message"]["content"] or ""
    if max_chars and max_chars > 0:
        return truncate(content, max_chars)
    return content.strip()


async def call_model(client: httpx.AsyncClient, disc: Discussion, model: ModelSpec, history: list) -> str:
    return await post_chat(
        client, disc, model, build_messages(disc, model, history),
        max_chars=disc.char_limit + DISPLAY_SLACK,
    )


def build_consensus_system_prompt(disc: Discussion, model: ModelSpec) -> str:
    others = [m.name for m in disc.models if m.id != model.id]
    others_list = ", ".join(others) if others else "(none)"
    return (
        f"You are {model.name}, one of several AI models who just finished a real-time think-tank "
        f"discussion. The other participants were: {others_list}. You are now collaborating with "
        f"them on a final wiki-style summary of that discussion: one model proposes a focused edit, "
        f"the others vote, accepted edits land in the article. Be substantive, balanced, and clear; "
        f"prefer surgical changes over wholesale rewrites.\n\n"
        f"TOPIC:\n{disc.topic}\n\n"
        f"EVIDENCE / CONTEXT:\n{disc.context or '(none provided)'}"
    )


def build_consensus_draft_messages(disc: Discussion, model: ModelSpec, wiki_cap: int) -> list[dict]:
    msgs = [{"role": "system", "content": build_consensus_system_prompt(disc, model)}]
    transcript = "\n\n".join(
        f"{('You' if m.model_id == model.id else m.model_name)}: {m.content}"
        for m in disc.history
    )
    reflections = "\n\n".join(
        f"{r.model_name} (shift {r.shift if r.shift is not None else '?'}): {r.content}"
        for r in disc.reflections
    ) or "(none)"
    msgs.append({
        "role": "user",
        "content": (
            f"The discussion and closing reflections are over. You have been chosen at random to "
            f"write the FIRST DRAFT of a collaborative wiki-style summary capturing what was discussed, "
            f"the strongest points raised, the consensus reached, and what (if anything) remains "
            f"contested. Other participants will then propose revisions one at a time and vote on them.\n\n"
            f"DISCUSSION TRANSCRIPT:\n\n{transcript}\n\n"
            f"CLOSING REFLECTIONS:\n\n{reflections}\n\n"
            f"Write a concise wiki-style article under {wiki_cap} characters. No preamble, no name "
            f"prefix — output the article body only."
        ),
    })
    return msgs


def build_consensus_revision_messages(disc: Discussion, model: ModelSpec, current: str, wiki_cap: int) -> list[dict]:
    msgs = [{"role": "system", "content": build_consensus_system_prompt(disc, model)}]
    msgs.append({
        "role": "user",
        "content": (
            f"The collaborative wiki-style summary of the discussion currently reads:\n\n"
            f"{current}\n\n"
            f"(end of current article — {len(current)} chars)\n\n"
            f"Propose ONE focused revision — a specific addition, deletion, rephrasing, or "
            f"correction. Aim for a surgical change (e.g., 'add a sentence on X', 'tighten the "
            f"second paragraph', 'drop the redundant clause'); do not rewrite the article wholesale. "
            f"Output the FULL article with your edit applied so the others can see it in context. "
            f"There is no strict character limit on the article — write as much as the topic warrants.\n\n"
            f"Reply in EXACTLY this format (no preamble, no code fence):\n\n"
            f"SUMMARY: <one short sentence describing exactly what you changed>\n"
            f"===ARTICLE===\n"
            f"<the FULL article with your edit applied>"
        ),
    })
    return msgs


def build_consensus_vote_messages(
    disc: Discussion, voter: ModelSpec, current: str, revised: str, proposer_name: str, summary: str,
) -> list[dict]:
    msgs = [{"role": "system", "content": build_consensus_system_prompt(disc, voter)}]
    msgs.append({
        "role": "user",
        "content": (
            f"The collaborative wiki article currently reads:\n\n"
            f"{current}\n\n"
            f"(end of current article)\n\n"
            f"@{proposer_name} proposes the following revision — summarised as: \"{summary}\".\n"
            f"Revised article:\n\n"
            f"{revised}\n\n"
            f"(end of proposed article)\n\n"
            f"Vote whether to apply the revision. Reply in EXACTLY this format "
            f"(no preamble, no code fence):\n\n"
            f"VOTE: yes\n"
            f"REASON: <one short sentence>\n\n"
            f"(replace `yes` with `no` or `abstain` as appropriate.)"
        ),
    })
    return msgs


async def consensus_propose(
    client: httpx.AsyncClient, disc: Discussion, model: ModelSpec, current: str, wiki_cap: int,
) -> tuple[str, str, str]:
    raw = await post_chat(
        client, disc, model,
        build_consensus_revision_messages(disc, model, current, wiki_cap),
        max_chars=0,
        max_tokens=4096,
    )
    summary, revised = parse_proposal(raw)
    return summary, revised, raw


async def consensus_vote(
    client: httpx.AsyncClient, disc: Discussion, voter: ModelSpec,
    current: str, revised: str, proposer_name: str, summary: str,
) -> tuple[str, str]:
    raw = await post_chat(
        client, disc, voter,
        build_consensus_vote_messages(disc, voter, current, revised, proposer_name, summary),
        max_chars=0,
        max_tokens=300,
    )
    vote, reason = parse_vote(raw)
    return vote, reason[:280]


def make_diff(before: str, after: str) -> list[dict]:
    a = (before or "").splitlines()
    b = (after or "").splitlines()
    matcher = difflib.SequenceMatcher(None, a, b)
    hunks = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in a[i1:i2]:
                hunks.append({"op": "=", "text": line})
        elif tag == "delete":
            for line in a[i1:i2]:
                hunks.append({"op": "-", "text": line})
        elif tag == "insert":
            for line in b[j1:j2]:
                hunks.append({"op": "+", "text": line})
        elif tag == "replace":
            for line in a[i1:i2]:
                hunks.append({"op": "-", "text": line})
            for line in b[j1:j2]:
                hunks.append({"op": "+", "text": line})
    return hunks


async def run_consensus(client: httpx.AsyncClient, disc: Discussion) -> None:
    spoken_ids = {m.model_id for m in disc.history}
    models = [m for m in disc.models if m.id in spoken_ids]
    if len(models) < 2:
        return
    n = len(models)
    threshold = n / 2
    wiki_cap = disc.char_limit * 3

    order = list(models)
    random.shuffle(order)

    emit(disc, {
        "type": "consensus_started",
        "models": [{"id": m.id, "name": m.name} for m in order],
        "threshold": threshold,
        "n": n,
        "wiki_cap": wiki_cap,
    })

    drafter = order[0]
    emit(disc, {
        "type": "consensus_proposing",
        "kind": "draft",
        "turn": 1,
        "model_id": drafter.id,
        "model_name": drafter.name,
    })
    started = time.monotonic()
    try:
        draft = await post_chat(
            client, disc, drafter,
            build_consensus_draft_messages(disc, drafter, wiki_cap),
            max_chars=wiki_cap,
            max_tokens=max(800, wiki_cap // 2),
        )
    except Exception as e:
        emit(disc, {
            "type": "consensus_failed",
            "stage": "draft",
            "model_id": drafter.id,
            "model_name": drafter.name,
            "error": str(e),
        })
        return

    disc.consensus_draft = draft
    record = {
        "turn": 1,
        "kind": "draft",
        "proposer_id": drafter.id,
        "proposer_name": drafter.name,
        "summary": "Initial draft",
        "revised": draft,
        "accepted": True,
        "yes": 0, "no": 0, "abstain": 0,
        "votes": [],
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }
    disc.consensus.append(record)
    save_transcript(disc)
    emit(disc, {"type": "consensus_turn", "turn_record": record, "draft": disc.consensus_draft})

    for i, proposer in enumerate(order[1:], start=2):
        emit(disc, {
            "type": "consensus_proposing",
            "kind": "revision",
            "turn": i,
            "model_id": proposer.id,
            "model_name": proposer.name,
        })
        turn_started = time.monotonic()
        prev_draft = disc.consensus_draft
        try:
            summary, revised, _raw = await consensus_propose(
                client, disc, proposer, prev_draft, wiki_cap,
            )
        except Exception as e:
            record = {
                "turn": i,
                "kind": "revision",
                "proposer_id": proposer.id,
                "proposer_name": proposer.name,
                "summary": f"(proposal failed: {e})",
                "revised": "",
                "diff": [],
                "accepted": False,
                "yes": 0, "no": 0, "abstain": 0,
                "votes": [],
                "elapsed_ms": int((time.monotonic() - turn_started) * 1000),
                "error": str(e),
            }
            disc.consensus.append(record)
            save_transcript(disc)
            emit(disc, {"type": "consensus_turn", "turn_record": record, "draft": disc.consensus_draft})
            continue

        voters = [m for m in models if m.id != proposer.id]
        diff = make_diff(prev_draft, revised)
        emit(disc, {
            "type": "consensus_voting",
            "turn": i,
            "proposer_id": proposer.id,
            "proposer_name": proposer.name,
            "summary": summary,
            "diff": diff,
            "revised": revised,
            "voters": [{"id": v.id, "name": v.name} for v in voters],
        })

        async def vote_and_emit(v: ModelSpec) -> dict:
            try:
                vote, reason = await consensus_vote(
                    client, disc, v, prev_draft, revised, proposer.name, summary,
                )
            except Exception as e:
                vote, reason = "abstain", f"error: {e}"
            rec = {"model_id": v.id, "model_name": v.name, "vote": vote, "reason": reason[:280]}
            emit(disc, {"type": "consensus_vote", "turn": i, "vote": rec})
            return rec

        votes = await asyncio.gather(*(vote_and_emit(v) for v in voters))
        yes_n = sum(1 for r in votes if r["vote"] == "yes")
        no_n = sum(1 for r in votes if r["vote"] == "no")
        abst_n = sum(1 for r in votes if r["vote"] == "abstain")

        accepted = (1 + yes_n) >= threshold
        if accepted:
            disc.consensus_draft = revised

        record = {
            "turn": i,
            "kind": "revision",
            "proposer_id": proposer.id,
            "proposer_name": proposer.name,
            "summary": summary,
            "revised": revised,
            "diff": diff,
            "accepted": accepted,
            "yes": yes_n, "no": no_n, "abstain": abst_n,
            "votes": votes,
            "elapsed_ms": int((time.monotonic() - turn_started) * 1000),
        }
        disc.consensus.append(record)
        save_transcript(disc)
        emit(disc, {"type": "consensus_turn", "turn_record": record, "draft": disc.consensus_draft})

    final_path = save_final_consensus(disc)
    emit(disc, {
        "type": "consensus_completed",
        "draft": disc.consensus_draft,
        "saved_to": final_path.name if final_path else None,
    })


async def run_discussion(disc: Discussion) -> None:
    """Each model runs its own loop, firing as soon as it's free and off cooldown.

    Slow responses are NOT cancelled - they land on the transcript whenever they
    return, even if other models have spoken in the meantime (so they're replying
    to a stale snapshot). After landing a message a model must wait `cooldown`
    other messages before firing again.
    """
    disc.status = "running"
    disc.started_at = datetime.datetime.now().isoformat(timespec="seconds")
    save_transcript(disc)
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
            save_transcript(disc)
            emit(disc, {
                "type": "message",
                "message": asdict(msg),
                "history_len_at_start": snapshot_len,
                "stale_by": (len(disc.history) - 1) - snapshot_len,
            })
            announce()

    async def reflect(client: httpx.AsyncClient, model: ModelSpec):
        emit(disc, {
            "type": "reflection_started",
            "model_id": model.id,
            "model_name": model.name,
        })
        started = time.monotonic()
        try:
            raw = await post_chat(
                client, disc, model, build_closing_messages(disc, model),
                max_chars=disc.char_limit + DISPLAY_SLACK + 30,
            )
        except Exception as e:
            emit(disc, {
                "type": "reflection_failed",
                "model_id": model.id,
                "model_name": model.name,
                "error": str(e),
            })
            return
        elapsed_ms = int((time.monotonic() - started) * 1000)
        shift, content = extract_shift(raw)
        refl = Message(
            turn=len(disc.reflections) + 1,
            model_id=model.id,
            model_name=model.name,
            content=content,
            elapsed_ms=elapsed_ms,
            shift=shift,
        )
        disc.reflections.append(refl)
        save_transcript(disc)
        emit(disc, {"type": "reflection", "message": asdict(refl)})

    try:
        async with httpx.AsyncClient() as client:
            tasks = [asyncio.create_task(speaker_loop(client, m)) for m in disc.models]
            await asyncio.gather(*tasks, return_exceptions=True)

            spoken_ids = {m.model_id for m in disc.history}
            reflectors = [m for m in disc.models if m.id in spoken_ids]
            if reflectors:
                emit(disc, {
                    "type": "closing_started",
                    "models": [{"id": m.id, "name": m.name} for m in reflectors],
                })
                await asyncio.gather(
                    *(reflect(client, m) for m in reflectors),
                    return_exceptions=True,
                )
            await run_consensus(client, disc)
        disc.status = "completed"
        emit(disc, {"type": "completed"})
    except Exception as e:
        disc.status = "error"
        emit(disc, {"type": "error", "message": str(e)})
    finally:
        disc.completed = True
        try:
            save_transcript(disc)
        except Exception:
            pass
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
