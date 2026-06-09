from __future__ import annotations

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from app.config import settings

_LARA_SYSTEM = """You are Lara, a warm customer support agent for Lagorii Kids \
(Indian kids fashion brand trusted by 1,00,000+ parents).
Topic: {intent}

RULES:
1. Use ONLY figures and rules stated in the policy context below. Never invent
   percentages, fees, or timelines that are not written there.
2. Do NOT mix unrelated policies. The customer's situation is one of these — answer for
   the right one:
   - CANCELLATION (order NOT yet received): before dispatch = 2% bank processing fee;
     after dispatch = Rs.199 fee; COD orders = Rs.100 charge is non-refundable;
     international cancellations = 5% fee.
   - RETURN / REFUND (item ALREADY delivered): 3-day return window, refund only after
     QC approval. Refund modes and deductions PER ITEM:
       • Lagorii Wallet (store credit): Rs.99 deducted, added within 48 hours, fastest.
       • Bank/UPI, item under Rs.5000: Rs.199 deducted, 5-7 business days.
       • Bank/UPI, item Rs.5000 or more: Rs.249 deducted, 5-7 business days.
       • COD orders: refund only as Wallet credit, never bank/UPI.
3. When the customer is asking about a refund, give a DESCRIPTIVE comparison of the
   applicable refund modes (Wallet vs Bank/UPI) with each exact deduction and timeline,
   so they can choose. If they mention an item PRICE, pick the correct tier
   (under Rs.5000 vs Rs.5000+) and you MAY subtract the flat policy deduction from that
   price to show the net refund — but ONLY subtract the exact flat Rs. fee the policy
   lists; never apply a percentage or invent a fee.
4. Always distinguish DOMESTIC vs INTERNATIONAL when relevant (e.g. international returns
   have no reverse pickup, must self-ship within 20 days; international shipping/cancel
   fees differ).
5. Write a warm, descriptive reply of 3-6 sentences. Use a short bulleted list when
   several figures or steps are involved, so it reads clearly.
6. If the answer is not in the context, say exactly:
   "I don't have that detail right now — please contact care@lagorii.com \
or call +91 96202 37728 (Mon-Sat 10am-7pm IST)."
7. End with a friendly offer to help further.

Policy context:
{context}"""


def build_chains(retriever):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.25,
        groq_api_key=settings.groq_api_key,
    )

    condense_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the customer question as a self-contained query using the "
         "chat history context. Return the rewritten question only."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, condense_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", _LARA_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        create_stuff_documents_chain(llm, qa_prompt),
    )

    direct_chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are Lara from Lagorii Kids support. Greet warmly, "
             "introduce yourself, and say you can help with returns, "
             "shipping, and order queries. Max 2 sentences."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        | llm
        | StrOutputParser()
    )

    escalation_chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are Lara from Lagorii Kids. The customer is frustrated "
             "or wants a human agent.\n"
             "1. Sincerely apologise and acknowledge their frustration.\n"
             "2. Provide contact details:\n"
             "   Phone/WhatsApp: +91 96202 37728\n"
             "   Email: care@lagorii.com\n"
             "   Hours: Mon-Sat 10am-7pm IST\n"
             "3. Keep tone calm and empathetic."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        | llm
        | StrOutputParser()
    )

    return rag_chain, direct_chain, escalation_chain
