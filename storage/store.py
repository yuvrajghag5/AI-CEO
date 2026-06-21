
import json
import os
import chromadb
from config.paths import CLEAN_DIR, VECTORDB
 
INPUT_FILE = CLEAN_DIR / "chunks.json"
CHROMA_DIR = VECTORDB / "./chroma_db"
COLLECTION_NAME = "ai_ceo_documents"
 
 
def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []
 
 
def main():
    chunks = load_json(INPUT_FILE)
    if not chunks:
        print(f"No chunks found in {INPUT_FILE}. Run chunks.py first.")
        return
 
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)
 
    existing = collection.get(include=[])  # only need ids, not metadata/documents
    already_stored_ids = set(existing["ids"])
 
    if already_stored_ids:
        print(f"Found existing collection with {len(already_stored_ids)} chunks already stored.")
    else:
        print("No existing data in ChromaDB collection — starting fresh.")
 
    new_chunks = [c for c in chunks if str(c["chunk_id"]) not in already_stored_ids]
    print(f"{len(chunks)} total chunks -> {len(new_chunks)} new to embed and store")
 
    if not new_chunks:
        print("Nothing new to store.")
        return
 
    ids = [str(c["chunk_id"]) for c in new_chunks]
    texts = [c["text"] for c in new_chunks]
    metadatas = [
        {
            "doc_id": c.get("doc_id"),
            "source": c.get("source") or "",
            "title": c.get("title") or "",
            "url": c.get("url") or "",
            "published_date": c.get("published_date") or "",
            "sentiment_label": c.get("sentiment_label") or "",
            "sentiment_score": c.get("sentiment_score") or 0.0,
        }
        for c in new_chunks
    ]
 
    collection.add(ids=ids, documents=texts, metadatas=metadatas)
 
    print(f"Inserted {len(ids)} new chunks.")
    print(f"Collection '{COLLECTION_NAME}' now contains {collection.count()} total chunks.")
 
 
if __name__ == "__main__":
    main()