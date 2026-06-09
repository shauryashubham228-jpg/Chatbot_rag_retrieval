"""
Per-session conversation memory: a simple sliding window of the last
WINDOW_SIZE turns (no summarization).
"""
from __future__ import annotations

from langchain_core.chat_history import InMemoryChatMessageHistory

_store: dict[str, InMemoryChatMessageHistory] = {}
WINDOW_SIZE = 6  # keep last 6 turns (12 messages)


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


def get_lc_history(session_id: str) -> list:
    msgs = get_history(session_id).messages
    return msgs[-(WINDOW_SIZE * 2):]


def save_turn(session_id: str, user_msg: str, ai_msg: str) -> None:
    h = get_history(session_id)
    h.add_user_message(user_msg)
    h.add_ai_message(ai_msg)


def clear_memory(session_id: str) -> None:
    if session_id in _store:
        _store[session_id].clear()
