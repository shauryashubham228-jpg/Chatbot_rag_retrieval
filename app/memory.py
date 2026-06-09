"""
Per-session conversation memory with a summary buffer.

- Keeps the last WINDOW_SIZE turns verbatim (a sliding window).
- Once older turns fall out of the window, they are folded into a single
  running concise summary that persists for the whole session.

So the LLM always sees:  [summary of earlier chat]  +  [last WINDOW_SIZE turns].
"""
from __future__ import annotations

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq

from app.config import settings

WINDOW_SIZE = 6  # turns kept verbatim (WINDOW_SIZE * 2 messages)

# session_id -> {"history": InMemoryChatMessageHistory, "summary": str}
_store: dict[str, dict] = {}

_summarizer = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    groq_api_key=settings.groq_api_key,
)


def _session(session_id: str) -> dict:
    if session_id not in _store:
        _store[session_id] = {"history": InMemoryChatMessageHistory(), "summary": ""}
    return _store[session_id]


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    return _session(session_id)["history"]


def get_lc_history(session_id: str) -> list:
    """Running summary (if any) as a SystemMessage + the last WINDOW_SIZE turns."""
    sess = _session(session_id)
    recent = sess["history"].messages[-(WINDOW_SIZE * 2):]
    if sess["summary"]:
        summary_msg = SystemMessage(
            content=f"Summary of earlier conversation with this customer: {sess['summary']}"
        )
        return [summary_msg] + recent
    return recent


def _summarize(previous_summary: str, overflow_msgs: list) -> str:
    lines = []
    for m in overflow_msgs:
        role = "Customer" if m.type == "human" else "Lara"
        lines.append(f"{role}: {m.content}")
    transcript = "\n".join(lines)
    prompt = (
        "You maintain a running summary of a customer-support chat for Lagorii Kids.\n"
        f"Existing summary (may be empty):\n{previous_summary or '(none)'}\n\n"
        f"New messages to fold in:\n{transcript}\n\n"
        "Return an updated, concise summary (3-4 sentences max) that preserves key "
        "facts: what the customer ordered/asked, whether the item was delivered, "
        "amounts/timelines discussed, and any unresolved request. Summary only."
    )
    try:
        return _summarizer.invoke(prompt).content.strip()
    except Exception:
        return previous_summary  # never break the chat on a summary failure


def save_turn(session_id: str, user_msg: str, ai_msg: str) -> None:
    sess = _session(session_id)
    hist = sess["history"]
    hist.add_user_message(user_msg)
    hist.add_ai_message(ai_msg)

    # Fold anything beyond the window into the running summary.
    overflow = hist.messages[:-(WINDOW_SIZE * 2)]
    if overflow:
        sess["summary"] = _summarize(sess["summary"], overflow)
        # keep only the last WINDOW_SIZE turns verbatim
        kept = hist.messages[-(WINDOW_SIZE * 2):]
        hist.clear()
        for m in kept:
            hist.add_message(m)


def clear_memory(session_id: str) -> None:
    if session_id in _store:
        _store[session_id]["history"].clear()
        _store[session_id]["summary"] = ""
