
import os
import chromadb

from dotenv import load_dotenv
from app.core.supabase_client import get_supabase_client

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_FOLDER = os.path.join(BASE_DIR, "vectorstore")

COLLECTION_NAME = "customer_knowledge_base"
BATCH_SIZE = 25


def migrate_collection():
    print("\n=== CHROMADB TO SUPABASE MIGRATION ===")

    # Read existing ChromaDB. Nothing is deleted or modified.
    chroma_client = chromadb.PersistentClient(path=VECTOR_FOLDER)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    total = collection.count()
    print("Source collection:", COLLECTION_NAME)
    print("Source chunks:", total)

    if total == 0:
        raise ValueError("Source collection is empty.")

    supabase = get_supabase_client()
    migrated = 0
    verified_ids = []

    for offset in range(0, total, BATCH_SIZE):
        data = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas", "embeddings"]
        )

        records = []

        for doc_id, document, metadata, embedding in zip(
            data["ids"],
            data["documents"],
            data["metadatas"],
            data["embeddings"]
        ):
            if not document or embedding is None:
                raise ValueError(f"Missing content or embedding: {doc_id}")

            if len(embedding) != 384:
                raise ValueError(f"Invalid embedding dimensions: {doc_id}")

            metadata = metadata or {}

            records.append({
                "id": doc_id,
                "role_scope": "CUSTOMER",
                "title": metadata.get(
                    "source_file", "Customer knowledge"
                ),
                "content": document,
                "source": metadata.get("source_file", "unknown"),
                "chunk_index": offset + len(records),
                "metadata": {
                    **metadata,
                    "chroma_original_id": doc_id,
                    "original_collection": COLLECTION_NAME
                },
                "embedding": [float(value) for value in embedding]
            })

            verified_ids.append(doc_id)

        # Upsert makes reruns safe: existing IDs are updated, not duplicated.
        if records:
            supabase.table("knowledge_documents").upsert(
                records,
                on_conflict="id"
            ).execute()

        migrated += len(records)
        print(f"Uploaded: {migrated}/{total}")

    # Verify every migrated original ID exists in Supabase.
    found_ids = set()

    for start in range(0, len(verified_ids), BATCH_SIZE):
        batch_ids = verified_ids[start:start + BATCH_SIZE]

        result = (
            supabase.table("knowledge_documents")
            .select("id")
            .in_("id", batch_ids)
            .execute()
        )

        found_ids.update(row["id"] for row in result.data)

    missing_ids = set(verified_ids) - found_ids

    print("\n=== MIGRATION VERIFICATION ===")
    print("Source chunks:", total)
    print("Uploaded chunks:", migrated)
    print("Verified chunks:", len(found_ids))
    print("Missing chunks:", len(missing_ids))

    if missing_ids:
        raise RuntimeError(
            f"Migration incomplete. Missing {len(missing_ids)} IDs."
        )

    print("\nSUCCESS: All original chunks migrated!")
    print("Original ChromaDB remains untouched.")


if __name__ == "__main__":
    migrate_collection()
