"""
FastAPI entry point — mounts Gradio UI at /  and exposes a REST /chat endpoint.
Run locally:  uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import os

import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

from app.agent import reset_session, run_agent
from app.gradio_ui import build_demo

app = FastAPI(title="Lagorii Kids — Lara Support Agent")


# ── REST endpoint (useful for programmatic access / testing) ────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    intent: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    result = run_agent(req.message, req.session_id)
    return ChatResponse(**result)


@app.post("/reset/{session_id}")
def reset(session_id: str):
    reset_session(session_id)
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Mount Gradio UI at root ─────────────────────────────────────────────────
demo = build_demo()
app = gr.mount_gradio_app(app, demo, path="/")


# ── Local dev shortcut ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
