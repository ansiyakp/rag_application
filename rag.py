import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY or not GROQ_API_KEY:
    raise ValueError("QDRANT_URL, QDRANT_API_KEY or GROQ_API_KEY is missing.")

COLLECTION = "pdf_knowledge"
MODEL = "llama-3.1-8b-instant"
TOP_K = 15
THRESHOLD = 0.60
MAX_CONTEXT = 8

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
groq = Groq(api_key=GROQ_API_KEY)


def retrieve(query):
    vector = embedder.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=TOP_K,
        with_payload=True
    ).points

    return [
        r for r in results
        if r.score >= THRESHOLD
    ][:MAX_CONTEXT]


def generate_answer(query):
    results = retrieve(query)

    if not results:
        return (
            "I couldn't find this information in the uploaded PDF documents.",
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
            f"PDF: {file}\nPAGE: {page}\nCONTENT:\n{text}"
        )
        documents.append(text)

        source = f"{file} (Page {page})"
        if source not in seen:
            citations.append(source)
            seen.add(source)

    if not context:
        return (
            "I couldn't find this information in the uploaded PDF documents.",
            [],
            []
        )

    prompt = f"""
Answer the question using ONLY the PDF context below.

Rules:
- Do not use outside knowledge.
- Do not guess or invent information.
- If the answer is not supported by the PDFs, say exactly:
I couldn't find this information in the uploaded PDF documents.

PDF CONTEXT:
{chr(10).join(context)}

QUESTION:
{query}
"""

    try:
        response = groq.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict PDF-only question answering assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        answer = response.choices[0].message.content.strip()

    except Exception as e:
        return f"Groq error: {e}", [], []

    if "couldn't find this information" in answer.lower():
        return (
            "I couldn't find this information in the uploaded PDF documents.",
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
            answer, citations, _ = generate_answer(question)

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



            
