from __future__ import annotations

import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer

from app.knowledge_base import POLICY_TEXT

FAISS_PATH = Path("faiss_index")

HEADERS = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()


def build_retriever():
    embeddings = LocalEmbeddings()

    if FAISS_PATH.exists():
        vs = FAISS.load_local(
            str(FAISS_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS)
        docs = splitter.split_text(POLICY_TEXT)
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(str(FAISS_PATH))

    return vs.as_retriever(search_kwargs={"k": 4})
