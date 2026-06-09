"""
Run LangSmith evaluation:   python -m eval.evaluate
"""
from __future__ import annotations

import time

from langsmith import Client
from langsmith.evaluation import EvaluationResult, evaluate
from langchain_groq import ChatGroq

from app.agent import run_agent
from app.config import settings

DATASET_NAME = "lagorii-policy-eval-v1"

GOLDEN_DATASET = [
    {"question": "How many days do I have to return a product?",
     "reference": "You have 3 days from delivery to raise a return request.",
     "intent": "returns_refunds"},
    {"question": "What is the deduction for a Lagorii Wallet refund?",
     "reference": "Rs.99 is deducted per item for the Lagorii Wallet (store credit) refund.",
     "intent": "returns_refunds"},
    {"question": "I paid with COD. Can I get a bank account refund?",
     "reference": "No. COD orders are refunded only as Lagorii Wallet store credits, not bank or UPI.",
     "intent": "returns_refunds"},
    {"question": "Can I return innerwear?",
     "reference": "No. Innerwear is non-returnable along with socks, hair accessories, personalised, and clearance items.",
     "intent": "returns_refunds"},
    {"question": "How long does delivery take within India?",
     "reference": "Orders ship within 2 working days and are delivered in 4–5 working days in India.",
     "intent": "shipping_delivery"},
    {"question": "Is shipping free on a Rs.1500 order?",
     "reference": "Yes. Indian orders above Rs.999 get free shipping, so Rs.1500 qualifies.",
     "intent": "shipping_delivery"},
    {"question": "My package arrived damaged. What should I do?",
     "reference": "Refuse the shipment, photograph the package and label, then contact +91 9620237728 or care@lagorii.com immediately.",
     "intent": "shipping_delivery"},
    {"question": "What fee applies if I cancel after dispatch?",
     "reference": "A Rs.199 cancellation fee applies for orders cancelled after dispatch.",
     "intent": "cancellations"},
    {"question": "How much is the bank refund deduction for a Rs.3000 item?",
     "reference": "Rs.199 per item is deducted for bank/UPI refunds on items under Rs.5000.",
     "intent": "returns_refunds"},
    {"question": "How do I return an international order?",
     "reference": "Email care@lagorii.com within 7 days. Self-ship the item to the warehouse within 20 days. No reverse pickup outside India.",
     "intent": "returns_refunds"},
]


def _upload_dataset(client: Client) -> None:
    try:
        ds = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Golden Q&A pairs for Lagorii Kids support agent eval",
        )
        client.create_examples(
            inputs=[{"question": e["question"], "intent": e["intent"]} for e in GOLDEN_DATASET],
            outputs=[{"answer": e["reference"]} for e in GOLDEN_DATASET],
            dataset_id=ds.id,
        )
        print(f"✅ Dataset '{DATASET_NAME}' created")
    except Exception as exc:
        if "already exists" in str(exc).lower():
            print(f"ℹ️  Reusing existing dataset '{DATASET_NAME}'")
        else:
            raise


def _build_judge():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        groq_api_key=settings.groq_api_key,
    )


def _score(judge, criteria: str, desc: str, prediction: str, question: str, reference: str = "") -> float:
    ref_line = f"Reference answer: {reference}\n" if reference else ""
    prompt = (
        f"You are an expert QA evaluator for a customer support chatbot.\n\n"
        f"Question: {question}\n"
        f"{ref_line}"
        f"Chatbot answer: {prediction}\n\n"
        f'Evaluate on "{criteria}": {desc}\n\n'
        f"Reply with ONLY a single integer from 1 (worst) to 10 (best)."
    )
    raw = judge.invoke(prompt).content.strip()
    time.sleep(2)
    try:
        return min(max(int(raw.split()[0]), 1), 10) / 10.0
    except Exception:
        return 0.5


_judge = _build_judge()


def correctness_evaluator(run, example) -> EvaluationResult:
    return EvaluationResult(
        key="correctness",
        score=_score(
            _judge, "correctness",
            "Is the answer factually correct vs the reference? Penalise wrong Rs. amounts, wrong timelines, missing key info.",
            prediction=run.outputs.get("answer", ""),
            question=example.inputs.get("question", ""),
            reference=example.outputs.get("answer", ""),
        ),
    )


def faithfulness_evaluator(run, example) -> EvaluationResult:
    return EvaluationResult(
        key="faithfulness",
        score=_score(
            _judge, "faithfulness",
            "Does the answer contain ONLY facts from Lagorii policy? Penalise invented fees, timelines, or rules.",
            prediction=run.outputs.get("answer", ""),
            question=example.inputs.get("question", ""),
        ),
    )


def helpfulness_evaluator(run, example) -> EvaluationResult:
    return EvaluationResult(
        key="helpfulness",
        score=_score(
            _judge, "helpfulness",
            "Is the answer friendly, concise, and actionable for a customer? Does it clearly address the question and offer to help further?",
            prediction=run.outputs.get("answer", ""),
            question=example.inputs.get("question", ""),
        ),
    )


def agent_target(inputs: dict) -> dict:
    result = run_agent(
        user_message=inputs["question"],
        session_id=f"eval-{inputs.get('intent', 'x')}-{abs(hash(inputs['question']))}",
    )
    time.sleep(8)
    return {"answer": result["answer"], "intent": result["intent"]}


def main():
    client = Client()
    _upload_dataset(client)

    print("🚀 Running evaluation (sequential, ~30 Groq calls)…")
    results = evaluate(
        agent_target,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator, faithfulness_evaluator, helpfulness_evaluator],
        experiment_prefix="lagorii-groq-llama",
        metadata={"model": "llama-3.1-8b-instant", "retriever": "all-MiniLM-L6-v2"},
        max_concurrency=1,
    )

    import pandas as pd

    rows = []
    for r in results:
        try:
            q   = r["example"].inputs.get("question", "")[:55] + "…"
            row = {"question": q}
            for fb in r.get("evaluation_results", {}).get("results", []):
                row[fb.key] = round(fb.score, 2) if fb.score is not None else None
            rows.append(row)
        except Exception:
            pass

    if rows:
        df  = pd.DataFrame(rows)
        num = df.select_dtypes("number").columns
        print("\n📊 Per-example scores\n")
        print(df.to_string(index=False))
        print("\n📈 Mean scores")
        for col, val in df[num].mean().round(3).items():
            bar = "█" * int(val * 20)
            print(f"   {col:<22} {val:.3f}  {bar}")
    else:
        print("ℹ️  No local results — check https://smith.langchain.com → Experiments")


if __name__ == "__main__":
    main()
