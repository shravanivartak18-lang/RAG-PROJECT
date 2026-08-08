import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from pypdf import PdfReader

DB_PATH = "knowledge.db"
PDF_PATH = "OPERATING-SYSTEMS.pdf"
RESPONSE_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_API_KEY = "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"
GEN_MODEL = "openai/gpt-oss-20b"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
TOP_K = 5

st.set_page_config(page_title="RAG Chatbot", layout="wide")


def create_database(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            content TEXT,
            embedding TEXT
        )
        """
    )
    conn.commit()


def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
    words = text.replace("\n", " ").split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def tokenize_text(text: str) -> List[str]:
    tokens = [token.strip(".,!?;:\"'()[]") for token in text.lower().split()]
    return [token for token in tokens if token]


def score_similarity(query: str, content: str) -> float:
    query_tokens = tokenize_text(query)
    content_tokens = tokenize_text(content)
    if not query_tokens or not content_tokens:
        return 0.0
    query_set = set(query_tokens)
    content_set = set(content_tokens)
    common = query_set.intersection(content_set)
    if not common:
        return 0.0
    return len(common) / (len(query_set) + len(content_set) - len(common))


def insert_chunk(conn: sqlite3.Connection, source: str, content: str) -> None:
    conn.execute(
        "INSERT INTO chunks (source, content, embedding) VALUES (?, ?, ?)",
        (source, content, json.dumps([])),
    )
    conn.commit()


def get_all_chunks(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    cursor = conn.execute("SELECT id, source, content, embedding FROM chunks")
    rows = []
    for row in cursor.fetchall():
        rows.append(
            {
                "id": row[0],
                "source": row[1],
                "content": row[2],
                "embedding": json.loads(row[3]) if row[3] else [],
            }
        )
    return rows


def dot_product(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def vector_norm(value: List[float]) -> float:
    return sum(x * x for x in value) ** 0.5


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    denom = vector_norm(a) * vector_norm(b)
    return dot_product(a, b) / denom if denom else -1.0


def find_top_matches(question: str, chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    scored = []
    for chunk in chunks:
        score = score_similarity(question, chunk["content"])
        scored.append({**chunk, "score": score})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def generate_answer(api_key: str, prompt: str) -> str:
    url = f"{RESPONSE_BASE_URL}/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GEN_MODEL,
        "input": prompt,
        "max_output_tokens": 512,
        "temperature": 0.2,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        if "output" in data and isinstance(data["output"], list):
            # Responses API returns a list of response items
            texts = []
            for item in data["output"]:
                if isinstance(item, dict) and item.get("type") == "message":
                    for content in item.get("content", []):
                        if isinstance(content, dict) and content.get("type") == "output_text":
                            texts.append(content.get("text", ""))
                elif isinstance(item, dict) and "text" in item:
                    texts.append(str(item["text"]))
            if texts:
                return "\n".join(texts)
        if "output_text" in data:
            return str(data["output_text"])
        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if isinstance(choice, dict):
                return str(choice.get("text", choice.get("message", "")))
    raise ValueError(f"Unexpected generate response format: {data}")


def build_prompt(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    context_text = "\n\n---\n\n".join(
        f"Source: {chunk['source']}\n{chunk['content']}" for chunk in context_chunks
    )
    prompt = (
        "You are a helpful assistant with access to the following operating systems reference material. "
        "Use only the information provided in the context to answer the question. If the answer is not in the text, say that you do not know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer concisely and clearly."
    )
    return prompt


def ingest_pdf(conn: sqlite3.Connection) -> Tuple[int, int]:
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    text = load_pdf_text(PDF_PATH)
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if not chunks:
        raise ValueError("No text found inside the PDF.")

    created = 0
    for chunk_text_value in chunks:
        insert_chunk(conn, os.path.basename(PDF_PATH), chunk_text_value)
        created += 1
    return len(chunks), created


def main() -> None:
    st.title("📘 Operating Systems RAG Chatbot")
    st.write(
        "Ask questions about the PDF document, and the app will retrieve relevant passages from the local database before sending the prompt to Groq."
    )

    api_key = DEFAULT_GROQ_API_KEY

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    create_database(conn)

    chunks = get_all_chunks(conn)
    if not chunks:
        with st.spinner("Reading PDF and building chunks automatically..."):
            try:
                total_chunks, stored = ingest_pdf(conn)
                st.success(f"Ingested {stored} chunks from the PDF.")
                chunks = get_all_chunks(conn)
            except Exception as exc:
                st.error(f"Automatic ingestion failed: {exc}")

    if not chunks:
        st.info("No embedded chunks found. Please check the app log or API key.")

    if chunks:
        question = st.text_input("Ask a question about the PDF", key="question_input")
        if st.button("Get Answer") and question.strip():
            with st.spinner("Retrieving relevant passages and calling Groq..."):
                try:
                    top_chunks = find_top_matches(question.strip(), chunks, TOP_K)
                    prompt = build_prompt(question.strip(), top_chunks)
                    answer = generate_answer(api_key, prompt)

                    st.subheader("Answer")
                    st.write(answer)

                    st.subheader("Top context snippets")
                    for idx, chunk in enumerate(top_chunks, start=1):
                        st.markdown(
                            f"**{idx}. Source:** {chunk['source']} — similarity {chunk['score']:.4f}"
                        )
                        st.write(chunk["content"])

                except Exception as exc:
                    st.error(f"Chatbot request failed: {exc}")

    conn.close()


if __name__ == "__main__":
    main()
