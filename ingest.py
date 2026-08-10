import os
import uuid
import fitz

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


# ---------------- CONFIG ----------------

load_dotenv()

PDF_FOLDER = "data"
COLLECTION_NAME = "pdf_knowledge"
MODEL_NAME = "all-MiniLM-L6-v2"

VECTOR_SIZE = 384
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300


# ---------------- ENV ----------------

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError(
        "QDRANT_URL or QDRANT_API_KEY is missing from .env"
    )

if not os.path.exists(PDF_FOLDER):
    raise FileNotFoundError(
        f"PDF folder not found: {PDF_FOLDER}"
    )


# ---------------- QDRANT ----------------

print("\nConnecting to Qdrant...")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("✓ Qdrant connected")


# ---------------- EMBEDDING MODEL ----------------

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("✓ Model loaded")


# ---------------- COLLECTION ----------------

collections = client.get_collections().collections

if any(c.name == COLLECTION_NAME for c in collections):
    print(f"Deleting old collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE
    )
)

print(f"✓ Collection ready: {COLLECTION_NAME}")


# ---------------- CHUNKING ----------------

def chunk_text(text):
    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(text), step):
        chunk = text[start:start + CHUNK_SIZE].strip()

        if chunk:
            chunks.append(chunk)

        if start + CHUNK_SIZE >= len(text):
            break

    return chunks


# ---------------- EMBEDDING ----------------

def create_embedding(text):
    return model.encode(
        text,
        normalize_embeddings=True
    ).tolist()


# ---------------- PDF FILES ----------------

pdf_files = [
    f for f in os.listdir(PDF_FOLDER)
    if f.lower().endswith(".pdf")
]

if not pdf_files:
    raise FileNotFoundError(
        f"No PDF files found in '{PDF_FOLDER}'"
    )


# ---------------- INGESTION ----------------

total_pdfs = 0
total_pages = 0
total_chunks = 0
failed_chunks = 0

print(f"\nFound {len(pdf_files)} PDF files.\n")


for pdf_file in pdf_files:

    pdf_path = os.path.join(
        PDF_FOLDER,
        pdf_file
    )

    print(f"Processing {pdf_file}")

    try:

        doc = fitz.open(pdf_path)
        total_pdfs += 1

        for page_index, page in enumerate(doc):

            page_number = page_index + 1
            total_pages += 1

            text = page.get_text().strip()

            if not text:
                continue

            chunks = chunk_text(text)

            print(
                f"  Page {page_number}: "
                f"{len(chunks)} chunk(s)"
            )

            for chunk_id, chunk in enumerate(chunks):

                try:

                    embedding = create_embedding(chunk)

                    point_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{pdf_file}_"
                            f"{page_number}_"
                            f"{chunk_id}"
                        )
                    )

                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "file": pdf_file,
                            "page": page_number,
                            "chunk_id": chunk_id
                        }
                    )

                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=[point]
                    )

                    total_chunks += 1

                except Exception as e:

                    failed_chunks += 1

                    print(
                        f"    ✗ Chunk "
                        f"{chunk_id + 1}: {e}"
                    )

        doc.close()

    except Exception as e:

        print(
            f"  ✗ Error: {e}"
        )


# ---------------- RESULT ----------------

print("\n" + "=" * 55)
print("INGESTION COMPLETE")
print("=" * 55)

print(f"PDF files processed : {total_pdfs}")
print(f"Pages processed     : {total_pages}")
print(f"Chunks uploaded     : {total_chunks}")
print(f"Failed chunks       : {failed_chunks}")
print(f"Qdrant collection   : {COLLECTION_NAME}")

print("=" * 55)

if failed_chunks == 0:
    print("\n✓ All PDF chunks uploaded successfully.")
else:
    print(
        f"\n⚠ {failed_chunks} chunks failed."
    )

print("\nNext step: python rag.py")