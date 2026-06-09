from __future__ import annotations

from langsmith import traceable

from app.chains import build_chains
from app.intent_router import Intent, RAG_INTENTS, build_router_chain, classify_intent
from app.memory import clear_memory, get_lc_history, save_turn
from app.vector_store import build_retriever

# Build once at import time (FAISS index cached to disk on first run)
_retriever     = build_retriever()
_router_chain  = build_router_chain()
_rag_chain, _direct_chain, _escalation_chain = build_chains(_retriever)


@traceable(name="intent-router", run_type="chain")
def _traced_classify(message: str) -> str:
    return classify_intent(_router_chain, message).value


@traceable(name="lagorii-agent", run_type="chain")
def run_agent(user_message: str, session_id: str = "default") -> dict:
    if not user_message.strip():
        return {"answer": "Please type your question! 😊", "intent": "empty", "session_id": session_id}

    intent_str   = _traced_classify(user_message)
    intent       = Intent(intent_str)
    chat_history = get_lc_history(session_id)

    if intent == Intent.ESCALATE_HUMAN:
        answer = _escalation_chain.invoke({"input": user_message, "chat_history": chat_history})
    elif intent == Intent.OUT_OF_SCOPE:
        answer = _direct_chain.invoke({"input": user_message, "chat_history": chat_history})
    else:
        result = _rag_chain.invoke({
            "input": user_message,
            "chat_history": chat_history,
            "intent": intent.value.replace("_", " ").title(),
        })
        answer = result["answer"]

    save_turn(session_id, user_message, answer)
    return {"answer": answer, "intent": intent.value, "session_id": session_id}


def reset_session(session_id: str) -> None:
    clear_memory(session_id)
