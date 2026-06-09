from __future__ import annotations

import uuid

import gradio as gr

from app.agent import reset_session, run_agent

INTENT_BADGES = {
    "returns_refunds"   : "🔄 Returns & Refunds",
    "shipping_delivery" : "🚚 Shipping & Delivery",
    "cancellations"     : "❌ Cancellations",
    "order_support"     : "📦 Order Support",
    "escalate_human"    : "🚨 Escalated to Human",
    "out_of_scope"      : "💬 General",
    "empty"             : "",
}

HEADER = """
<div style="background:linear-gradient(135deg,#ff6b9d,#c44dff);
            padding:22px 28px;border-radius:14px;margin-bottom:10px;
            color:white;font-family:'Segoe UI',sans-serif;">
  <h1 style='margin:0;font-size:1.7rem;'>🛍️ D2C Chatbot— Support Chat</h1>
  <p style='margin:5px 0 0;font-size:.95rem;opacity:.93;'>
    Hi! I'm <strong>Lara</strong> — ask me about
    <strong>returns · refunds · shipping · cancellations · orders</strong>.
  </p>
</div>
"""

EXAMPLES = [
    "How do I return a product?",
    "What is the wallet refund deduction?",
    "I paid COD — can I get a bank refund?",
    "When will my order arrive?",
    "Is shipping free on a Rs.800 order?",
    "My package arrived damaged",
    "Cancel after dispatch — any fee?",
    "I want to talk to a human agent",
]


def chat_fn(user_msg: str, history: list, session_id: str):
    if not user_msg.strip():
        return "", history, session_id
    result = run_agent(user_msg, session_id)
    badge  = INTENT_BADGES.get(result["intent"], "")
    reply  = result["answer"] + (f"\n\n`{badge}`" if badge else "")
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": reply})
    return "", history, session_id


def reset_fn(session_id: str):
    reset_session(session_id)
    return [], str(uuid.uuid4())


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Lagorii Kids — Lara") as demo:
        session_id = gr.State(str(uuid.uuid4()))

        gr.HTML(HEADER)
        chatbot = gr.Chatbot(height=460)
        with gr.Row():
            msg  = gr.Textbox(
                placeholder="Ask about returns, shipping, refunds...",
                show_label=False,
                scale=5,
            )
            send = gr.Button("Send ➤", variant="primary", scale=1)

        gr.Examples(examples=EXAMPLES, inputs=msg, label="💡 Try these")
        clear = gr.Button("🗑️ New conversation", variant="secondary")
        gr.Markdown(
            "---\n"
            "📞 **+91 96202 37728** | ✉️ **care@lagorii.com** | "
            "🕐 Mon–Sat 10am–7pm IST"
        )

        send.click(chat_fn,  [msg, chatbot, session_id], [msg, chatbot, session_id])
        msg.submit(chat_fn,  [msg, chatbot, session_id], [msg, chatbot, session_id])
        clear.click(reset_fn, [session_id], [chatbot, session_id])

    return demo
