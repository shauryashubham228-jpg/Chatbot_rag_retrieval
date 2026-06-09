# 🛍️ D2C chatbot — Lara AI Support Agent

> An intelligent customer support chatbot for **Lagorii Kids** — an Indian kids fashion brand trusted by 1,00,000+ parents. Built with LangChain, Groq (Llama 3.3 70B), FAISS, and Gradio. Deployed on Railway.

---

## 🔗 Live Demo

**[→ Try Lara on Railway](https://chatbotragretrieval-production.up.railway.app/)**

---

## 🤖 What Lara Can Do

| Topic | Example Questions |
|---|---|
| 🔄 Returns & Refunds | "How do I return a product?" · "How much is deducted for wallet refund?" |
| 🚚 Shipping & Delivery | "Is shipping free on Rs.800 order?" · "When will my order arrive?" |
| ❌ Order Cancellations | "Can I cancel after dispatch?" · "What is the COD cancellation policy?" |
| 📦 Order Support | "I received a wrong item" · "Is COD available for my order?" |
| 🚨 Human Escalation | Detects frustrated customers and provides direct contact details |
| 💬 General Queries | Greetings, store info, customization policy, gift card redemption |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER MESSAGE                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTENT ROUTER                                │
│         llama-3.3-70b-versatile  │  temperature = 0.0          │
│                                                                 │
│   returns_refunds  │  shipping_delivery  │  cancellations       │
│   order_support    │  escalate_human     │  out_of_scope        │
└──────┬──────────────────┬──────────────────────┬───────────────┘
       │                  │                      │
       ▼                  ▼                      ▼
  RAG CHAIN          DIRECT CHAIN        ESCALATION CHAIN
  (policy Q&A)       (greetings /        (frustrated customer /
                      small talk)         wants human agent)
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   HISTORY-AWARE RAG CHAIN                       │
│                                                                 │
│  Step 1 — CONDENSE                                              │
│  Sliding window (last 6 turns) + current question               │
│  → Rephrased as self-contained standalone query                 │
│                                                                 │
│  Step 2 — RETRIEVE                                              │
│  FAISS vector store (sentence-transformers all-MiniLM-L6-v2)   │
│  → Top 4 most relevant policy chunks (k=4)                      │
│                                                                 │
│  Step 3 — ANSWER                                                │
│  llama-3.3-70b-versatile  │  temperature = 0.25                 │
│  Policy context + chat history + intent → final answer          │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              MEMORY  (Sliding Window — 6 turns)                 │
│         Save turn → trimmed to last 12 messages                 │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│               GRADIO UI  (mounted on FastAPI)                   │
│        Intent badge · Example prompts · Reset session           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
chatbot/
├── main.py                  # FastAPI app — mounts Gradio at / and exposes REST API
├── Dockerfile               # Railway deployment
├── requirements.txt
├── .env.example             # Environment variable template
├── inspect_retrieval.py     # Dev tool — inspect FAISS chunks for any query
│
├── app/
│   ├── config.py            # Pydantic settings (loads .env)
│   ├── knowledge_base.py    # Full policy text (single source of truth)
│   ├── vector_store.py      # Builds / loads FAISS index (cached to faiss_index/)
│   ├── intent_router.py     # 6-class intent classifier (Groq llama-3.3-70b)
│   ├── chains.py            # RAG chain + direct chain + escalation chain
│   ├── memory.py            # Sliding window memory (last 6 turns per session)
│   ├── agent.py             # Orchestrator — routes message → chain → save turn
│   └── gradio_ui.py         # Gradio Blocks UI with intent badges
│
└── eval/
    └── evaluate.py          # LangSmith evaluation — 10 golden Q&A pairs
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | [Groq](https://groq.com) — `llama-3.3-70b-versatile` (answer + router) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, no API needed) |
| Vector Store | FAISS (CPU) — cached to disk on first startup |
| Framework | LangChain 1.x (`langchain-classic`, `langchain-core`) |
| Memory | Sliding window — last 6 conversation turns per session |
| UI | Gradio 6.17.3 mounted on FastAPI |
| Tracing | LangSmith — every chain call traced end-to-end |
| Deployment | Railway (Docker) |

---

## 🚀 Run Locally

### 1. Clone and install

```bash
git clone https://github.com/shauryashubham228-jpg/lagorii-chatbot.git
cd lagorii-chatbot
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

```env
GROQ_API_KEY=your_groq_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=lagorii-agent
LANGCHAIN_TRACING_V2=true
```

### 3. Start the server

```bash
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — Lara is live.

> On first startup FAISS builds the index from `knowledge_base.py` and saves it to `faiss_index/`. Subsequent starts load from cache.

---

## 🔑 API Keys

| Key | Where to get |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) — free tier |
| `LANGSMITH_API_KEY` | [smith.langchain.com](https://smith.langchain.com) — free tier |

---

## 🌐 Deploy on Railway

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub**
3. Add environment variables in Railway dashboard (same as `.env`)
4. Railway auto-detects the `Dockerfile` and deploys
5. Delete `faiss_index/` from repo before pushing — Railway rebuilds the index on each deploy

---

## 🔌 REST API

The FastAPI server also exposes a REST endpoint for programmatic access:

```bash
POST /chat
{
  "message": "How do I return a COD order?",
  "session_id": "user-123"
}
```

```bash
POST /reset/{session_id}   # Clear conversation history
GET  /health               # Health check
```

---

## 🔍 Inspect Retrieval (Dev Tool)

See exactly which policy chunks FAISS retrieves for any query:

```bash
python inspect_retrieval.py "bank refund for item over Rs.5000"
python inspect_retrieval.py "can I cancel COD after dispatch"
```

---

## 📊 Evaluation

Run LangSmith evaluation with 10 golden Q&A pairs and 3 LLM-as-judge evaluators (correctness, faithfulness, helpfulness):

```bash
python -m eval.evaluate
```

Results appear in your LangSmith dashboard.

---

## 🧠 Key Design Decisions

**Why not LangGraph?**
The current architecture (intent router → chain dispatch → RAG) already handles conditional routing cleanly. LangGraph adds value for multi-step tool-calling agents — unnecessary complexity for a policy Q&A bot.

**Why FAISS over a hosted vector DB?**
Zero cost, zero latency, no API key needed. Policy text is small (~3KB) and static — a local index is rebuilt in under 2 seconds on every deploy.

**Why sliding window over summary memory?**
Simple, predictable, and fast. Summaries add an extra LLM call and can drift from the original facts. For a support bot where each query is mostly independent, 6-turn window is sufficient context.

**Why separate condense + answer steps?**
The condense step rephrases follow-up questions ("and for that item?") into self-contained queries before hitting FAISS. This dramatically improves retrieval accuracy on multi-turn conversations.

---

## 📸 Screenshots

> *Evaluation on Dataset 
(<img width="899" height="355" alt="image" src="https://github.com/user-attachments/assets/c4175b3f-30de-4165-89a9-95230de6f04c" />
chatbot ui
<img width="900" height="1600" alt="image" src="https://github.com/user-attachments/assets/e52bc267-5570-484e-b079-01c7bfbe3bb1" />
prompts
<img width="900" height="1600" alt="image" src="https://github.com/user-attachments/assets/74f23101-eca0-49a5-b740-a6bf35c74905" />


)*

---

## 👤 Author

Built by **[Shaurya Shubham]**
[LinkedIn](https://www.linkedin.com/in/shaurya-shubham-088a91215/) · [GitHub](https://github.com/shauryashubham228-jpg)

---

