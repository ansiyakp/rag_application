import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama

load_dotenv()


COLLECTION_NAME = "pdf_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 15
SIMILARITY_THRESHOLD = 0.55
MAX_CONTEXT_CHUNKS = 6

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("Qdrant credentials are missing from .env")


qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


def retrieve(query):

    query_vector = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
        with_vectors=False
    )

    relevant = []

    for point in results.points:

        score = point.score

        if score < SIMILARITY_THRESHOLD:
            continue

        payload = point.payload or {}

        text = payload.get("text", "")

        if not text.strip():
            continue

        relevant.append({
            "text": text,
            "file": payload.get("file", "Unknown"),
            "page": payload.get("page", "Unknown"),
            "score": score
        })

    relevant.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return relevant[:MAX_CONTEXT_CHUNKS]



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

    for i, item in enumerate(results):

        context.append(
            f"""
SOURCE {i + 1}
File: {item["file"]}
Page: {item["page"]}

Content:
{item["text"]}
"""
        )

    context_text = "\n".join(context)

    prompt = f"""
You are a strict PDF question-answering assistant.

Use ONLY the information contained in the PDF context below.

Rules:
- Do not use outside knowledge.
- Do not use internet knowledge.
- Do not guess.
- Do not assume missing information.
- Answer only when the context supports the answer.
- If the answer is not supported by the context, respond exactly:

I couldn't find this information in the uploaded PDF documents.

Keep the answer clear and concise.

PDF CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:
"""

    response = ollama.generate(
        model="gemma3:1b",
        prompt=prompt,
        options={
            "temperature": 0
        }
    )

    answer = response["response"].strip()

    not_found = (
        "I couldn't find this information in the "
        "uploaded PDF documents."
    )

    if not_found.lower() in answer.lower():

        return (
            not_found,
            [],
            []
        )

    citations = []
    seen = set()
    documents = []

    for item in results:

        source = (
            f'{item["file"]} '
            f'(Page {item["page"]})'
        )

        if source not in seen:

            citations.append(source)
            seen.add(source)

        documents.append(item["text"])

    return (
        answer,
        citations,
        documents
    )



if __name__ == "__main__":

    while True:

        question = input(
            "\nAsk a question (exit to quit): "
        ).strip()

        if question.lower() == "exit":
            print("\nExiting...")
            break

        if not question:
            print("\nPlease enter a question.")
            continue

        print("\nSearching your PDF documents...")

        try:

            answer, citations, docs = generate_answer(
                question
            )

            print("\n" + "=" * 50)
            print("ANSWER")
            print("=" * 50)

            print(answer)

            print("\n" + "=" * 50)
            print("SOURCES")
            print("=" * 50)

            if citations:

                for citation in citations:
                    print("-", citation)

            else:
                print("No relevant PDF sources found.")

        except Exception as e:

            print("\nERROR:", e)



            
