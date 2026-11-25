
import uuid

import chromadb


def main():
    client = chromadb.PersistentClient(path="vectorstore")

    collection = client.get_collection(
        "customer_knowledge_base"
    )

    ids = collection.get()["ids"]

    invalid_ids = []

    for item in ids:
        try:
            uuid.UUID(item)
        except (ValueError, TypeError, AttributeError):
            invalid_ids.append(item)

    print("Total ChromaDB IDs:", len(ids))
    print("Valid UUIDs:", len(ids) - len(invalid_ids))
    print("Invalid UUIDs:", len(invalid_ids))
    print("Sample IDs:", ids[:3])

    if invalid_ids:
        print("Invalid ID examples:", invalid_ids[:5])


if __name__ == "__main__":
    main()
