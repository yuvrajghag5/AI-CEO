
import json
import os
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from config.paths import CLEAN_DIR
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
INPUT_FILE = CLEAN_DIR / "sentiment_analysis.json"
OUTPUT_FILE = CLEAN_DIR / "chunks.json"
 
CHUNK_SIZE = CHUNK_SIZE
CHUNK_OVERLAP = CHUNK_OVERLAP
 
 
def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []
 
 
def main():
    documents = load_json(INPUT_FILE)
    if not documents:
        print(f"No documents found in {INPUT_FILE}. Run sentiment.py first.")
        return
 
    existing_chunks = load_json(OUTPUT_FILE)
    already_chunked_doc_ids = {chunk["doc_id"] for chunk in existing_chunks}
 
    if existing_chunks:
        print(f"Found existing {OUTPUT_FILE} with chunks from {len(already_chunked_doc_ids)} documents.")
    else:
        print(f"No existing {OUTPUT_FILE} found — creating it from scratch.")
 
    new_documents = [doc for doc in documents if doc.get("doc_id") not in already_chunked_doc_ids]
    print(f"{len(documents)} total documents -> {len(new_documents)} new to chunk")
 
    if not new_documents:
        print("Nothing new to chunk.")
        return
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
 
    max_existing_chunk_id = max((c.get("chunk_id", 0) for c in existing_chunks), default=0)
    next_chunk_id = max_existing_chunk_id + 1
 
    new_chunks = []
    for doc in new_documents:
        content = doc.get("content", "")
        if not content:
            continue
 
        pieces = splitter.split_text(content)
        for piece in pieces:
            new_chunks.append({
                "chunk_id": next_chunk_id,
                "doc_id": doc.get("doc_id"),
                "source": doc.get("source"),
                "title": doc.get("title"),
                "url": doc.get("url"),
                "published_date": doc.get("published_date"),
                "sentiment_label": doc.get("sentiment_label"),
                "sentiment_score": doc.get("sentiment_score"),
                "text": piece,
            })
            next_chunk_id += 1
 
    combined_chunks = existing_chunks + new_chunks
 
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(combined_chunks, f, indent=2, ensure_ascii=False)
 
    print(f"Created {len(new_chunks)} new chunks from {len(new_documents)} documents "
          f"(chunk_id {max_existing_chunk_id + 1} to {next_chunk_id - 1}).")
    print(f"Total chunks in {OUTPUT_FILE}: {len(combined_chunks)}")
 
 
if __name__ == "__main__":
    main()
 