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
Classify the customer message into EXACTLY ONE label.

KEY DISTINCTION — read carefully:
- cancellations  = the customer wants to STOP an order that has NOT been received yet
  (cancel before dispatch, cancel after dispatch, cancellation fees, the 2% bank
  processing fee, the Rs.199 after-dispatch fee, COD Rs.100 charge on cancellation,
  international 5% cancellation fee). Trigger words: "cancel", "stop my order",
  "don't want it anymore", "cancel before it ships".
- returns_refunds = the customer ALREADY RECEIVED the item and wants to send it back
  or get money back (return window of 3 days, QC approval, wallet refund Rs.99,
  bank/UPI refund Rs.199 or Rs.249, non-returnable items, how refunds are paid).
  Trigger words: "return", "refund", "send it back", "money back", "deduction".

Labels:
returns_refunds    – returning a delivered product, refund amounts/methods, wallet credits, QC, non-returnable items
shipping_delivery  – shipping costs, delivery timelines, tracking, damaged/tampered packages
cancellations      – cancelling/stopping an order, cancellation fees (before vs after dispatch)
order_support      – wrong item received, unpacking video, gift card orders, and COD
                     handling: the Rs.100 COD charge is an upfront, non-refundable fee,
                     and COD orders can ONLY be refunded as Lagorii Wallet credit (never
                     bank/UPI). Route COD-fee or COD-refund-method questions here.
escalate_human     – angry/frustrated customer, wants a manager or human agent
out_of_scope       – greetings, small talk, unrelated questions

If the message mentions BOTH cancelling and refund, choose cancellations only if the
order has not been received yet; otherwise choose returns_refunds.

Reply with ONLY the label. No explanation. No punctuation."""),
    ("human", "Customer message: {message}"),
])


def build_router_chain():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
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
