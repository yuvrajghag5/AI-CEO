
import chromadb
from langchain_ollama import ChatOllama
from prompt import RAG_PROMPT
from config.settings import TOP_K, OLLAMA_MODEL
from config.paths import VECTORDB
 
CHROMA_DIR = VECTORDB / "./chroma_db"
COLLECTION_NAME = "ai_ceo_documents"
OLLAMA_MODEL = OLLAMA_MODEL   # <-- match whatever you pulled via `ollama pull`
TOP_K = TOP_K
 
# Simple in-session memory: list of {"question": ..., "answer": ...} dicts.
# Not persisted to disk — resets every time the script restarts.
# Significance: lets follow-up questions reference earlier answers,
# instead of every question being treated as the first one ever asked.
memory = []
 
 
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)
 
 
def retrieve_context(collection, question, top_k=TOP_K):
    results = collection.query(query_texts=[question], n_results=top_k)
 
    chunks = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        text = results["documents"][0][i]
        chunks.append(f"[{meta.get('source')} | {meta.get('title')}]\n{text}")
 
    return "\n\n---\n\n".join(chunks)
 
 
def format_history(memory, max_turns=3):
    """Use only the last few turns to keep the prompt short."""
    if not memory:
        return "(no previous conversation)"
    recent = memory[-max_turns:]
    return "\n".join(f"Q: {turn['question']}\nA: {turn['answer']}" for turn in recent)
 
 
def answer_question(question, llm, collection):
    context = retrieve_context(collection, question)
    history = format_history(memory)
 
    prompt_text = RAG_PROMPT.format(history=history, context=context, question=question)
    response = llm.invoke(prompt_text)
    answer = response.content.strip()
 
    memory.append({"question": question, "answer": answer})
    return answer
 
 
def main():
    collection = get_collection()
    if collection.count() == 0:
        print("ChromaDB collection is empty. Run store.py first.")
        return
 
    llm = ChatOllama(model=OLLAMA_MODEL)
 
    print("RAG chat ready. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
 
        answer = answer_question(question, llm, collection)
        print(f"\nAgent: {answer}\n")
 
 
if __name__ == "__main__":
    main()