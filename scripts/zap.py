import sys

from dotenv import load_dotenv
from google.cloud import firestore

load_dotenv()


def delete_collection(coll_ref, batch_size):
    docs = coll_ref.limit(batch_size).stream()
    deleted = 0

    for doc in docs:
        print(f"Deleting doc {doc.id}")
        doc.reference.delete()
        deleted += 1

    if deleted >= batch_size:
        return delete_collection(coll_ref, batch_size)


def delete_document_and_subcollections(doc_ref):
    # Delete all subcollections first
    for subcollection in doc_ref.collections():
        for doc in subcollection.stream():
            delete_document_and_subcollections(doc.reference)
    # Delete the main document
    doc_ref.delete()
    print(f"Deleted: {doc_ref.path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python zap.py <collection_name>")
        sys.exit(1)

    collection_name = sys.argv[1]
    db = firestore.Client()
    collection_ref = db.collection(collection_name)
    delete_collection(collection_ref, 200)
