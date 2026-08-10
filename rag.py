import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama

load_dotenv()

# Config
COLLECTION = "pdf_knowledge"
MODEL = "gemma3:1b"
TOP_K = 10
THRESHOLD = 0.60
MAX_CONTEXT = 6

# Environment
url = os.getenv("QDRANT_URL")
key = os.getenv("QDRANT_API_KEY")

if not url or not key:
    raise ValueError("QDRANT_URL or QDRANT_API_KEY is missing")

# Connections
qdrant = QdrantClient(
    url=url,
    api_key=key,
    timeout=60
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query):
    vector = embedder.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=TOP_K,
        with_payload=True,
        with_vectors=False
    ).points

    return [
        r for r in results
        if r.score >= THRESHOLD
    ][:MAX_CONTEXT]


def generate_answer(query):
    results = retrieve(query)

    if not results:
        return (
            "I couldn't find this information in the "
            "uploaded PDF documents.",
            [],
            []
        )

    context = []
    citations = []
    documents = []
    seen = set()

    for r in results:
        data = r.payload or {}
        text = data.get("text", "")
        file = data.get("file", "Unknown file")
        page = data.get("page", "Unknown page")

        if not text:
            continue

        context.append(
            f"PDF: {file}\n"
            f"Page: {page}\n"
            f"Content:\n{text}"
        )

        documents.append(text)

        source = f"{file} (Page {page})"

        if source not in seen:
            citations.append(source)
            seen.add(source)

    if not context:
        return (
            "I couldn't find this information in the "
            "uploaded PDF documents.",
            [],
            []
        )

    prompt = f"""
You are a strict PDF question-answering assistant.

Answer ONLY from the PDF context below.

Rules:
- Do not use outside knowledge.
- Do not use internet knowledge.
- Do not guess.
- Do not invent information.
- Every factual statement must be supported by the PDFs.
- If the answer is not supported, reply exactly:

I couldn't find this information in the uploaded PDF documents.

PDF CONTEXT:
{"".join(context)}

QUESTION:
{query}

ANSWER:
"""

    try:
        response = ollama.generate(
            model=MODEL,
            prompt=prompt,
            options={"temperature": 0}
        )

        answer = response["response"].strip()

    except Exception as e:
        return f"Ollama error: {e}", [], []

    if "couldn't find this information" in answer.lower():
        return (
            "I couldn't find this information in the "
            "uploaded PDF documents.",
            [],
            []
        )

    return answer, citations, documents


if __name__ == "__main__":

    print("\n📚 PDF RAG Chatbot")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("Ask a question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a question.")
            continue

        print("\nSearching your PDF documents...")

        try:
            answer, citations, docs = generate_answer(question)

            print("\nANSWER\n")
            print(answer)

            print("\nSOURCES\n")

            if citations:
                for citation in citations:
                    print("-", citation)
            else:
                print("No relevant PDF source found.")

        except Exception as e:
            print("\nERROR:", e)


            