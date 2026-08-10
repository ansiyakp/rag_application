import os
import fitz
import chromadb
import ollama

PDF_FOLDER = "data"

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="pdf_knowledge"
)


def chunk_text(text, chunk_size=2000, overlap=300):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


def extract_page_text(page):

    return page.get_text()


for pdf_file in os.listdir(PDF_FOLDER):

    if not pdf_file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(
        PDF_FOLDER,
        pdf_file
    )

    print(f"Processing {pdf_file}")

    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):

        page = doc[page_num]

        text = extract_page_text(page)

        if not text.strip():
            continue

        chunks = chunk_text(text)

        for chunk_id, chunk in enumerate(chunks):

            try:

                embedding = ollama.embeddings(
                    model="nomic-embed-text",
                    prompt=chunk
                )["embedding"]

                unique_id = (
                    f"{pdf_file}"
                    f"_p{page_num}"
                    f"_c{chunk_id}"
                )

                collection.add(
                    ids=[unique_id],
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{
                        "file": pdf_file,
                        "page": page_num + 1
                    }]
                )

            except Exception as e:

                print(
                    f"Error embedding page "
                    f"{page_num + 1}: {e}"
                )

print("\nIngestion Complete")

