import chromadb
import ollama

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "pdf_knowledge"
)


def retrieve(query, k=15):

    query_embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=query
    )["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results


def generate_answer(query):

    results = retrieve(query)

    documents = results["documents"][0]
    metadata = results["metadatas"][0]

    context = "\n\n".join(documents)

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the supplied context.

If the answer is partially present,
provide the best possible answer.

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.generate(
        model="gemma3:1b",
        prompt=prompt
    )

    answer = response["response"]

    citations = []

    seen = set()

    for meta in metadata:

        source = (
            f"{meta['file']} "
            f"(Page {meta['page']})"
        )

        if source not in seen:

            citations.append(source)

            seen.add(source)

    return (
        answer,
        citations,
        documents
    )


if __name__ == "__main__":

    while True:

        question = input(
            "\nAsk a question (exit to quit): "
        )

        if question.lower() == "exit":
            break

        answer, citations, docs = generate_answer(
            question
        )

        print("\nANSWER\n")

        print(answer)

        print("\nSOURCES\n")

        for c in citations:
            print("-", c)

            