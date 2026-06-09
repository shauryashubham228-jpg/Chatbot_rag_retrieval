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
1. Answer ONLY using exact figures copied verbatim from the policy context below.
2. NEVER calculate, derive, estimate, or give example math. Do NOT compute percentages
   or subtract fees. State the exact fee/amount/timeline the policy lists, nothing more.
3. Do NOT mix policies: cancellation fees, wallet refunds, bank/UPI refunds, and COD
   are separate. Use only the clause that matches the customer's exact question.
4. If the answer is not in the context, say exactly:
   "I don't have that detail right now — please contact care@lagorii.com \
or call +91 96202 37728 (Mon-Sat 10am-7pm IST)."
5. Be concise, friendly, and specific (quote exact Rs. amounts and timelines).
6. End with a short offer to help further.

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
