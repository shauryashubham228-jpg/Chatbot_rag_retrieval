from __future__ import annotations

from enum import Enum

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.config import settings


class Intent(str, Enum):
    RETURNS_REFUNDS   = "returns_refunds"
    SHIPPING_DELIVERY = "shipping_delivery"
    CANCELLATIONS     = "cancellations"
    ORDER_SUPPORT     = "order_support"
    ESCALATE_HUMAN    = "escalate_human"
    OUT_OF_SCOPE      = "out_of_scope"


RAG_INTENTS = {
    Intent.RETURNS_REFUNDS,
    Intent.SHIPPING_DELIVERY,
    Intent.CANCELLATIONS,
    Intent.ORDER_SUPPORT,
}

_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intent classifier for Lagorii Kids customer support.
Classify the customer message into EXACTLY ONE of these labels:
returns_refunds    – returning products, refund amounts, wallet credits, QC
shipping_delivery  – shipping costs, delivery timelines, tracking, damaged packages
cancellations      – cancelling an order, cancellation fees
order_support      – wrong item, unpacking video, COD charges, gift cards
escalate_human     – angry/frustrated customer, wants manager or human agent
out_of_scope       – greetings, small talk, unrelated questions
Reply with ONLY the label. No explanation. No punctuation."""),
    ("human", "Customer message: {message}"),
])


def build_router_chain():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        groq_api_key=settings.groq_api_key,
    )
    return _ROUTER_PROMPT | llm | StrOutputParser()


def classify_intent(chain, message: str) -> Intent:
    raw = chain.invoke({"message": message}).strip().lower()
    try:
        return Intent(raw)
    except ValueError:
        return Intent.OUT_OF_SCOPE
